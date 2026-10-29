use clap::{Parser, Subcommand};
use image::{DynamicImage, ImageReader};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

mod phash;
use crate::phash::PerceptualHash;
use phash::{hamming_distance, perceptual_hash};

#[derive(Parser, Debug)]
#[command(version, about = "Find and group near-duplicate images")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Find and report near-duplicate images.
    Scan {
        /// Directory to scan
        #[arg(default_value = ".")]
        directory: PathBuf,

        /// Scan subdirectories recursively.
        #[arg(short, long)]
        recursive: bool,

        /// Lists similar images as sets.
        #[arg(short, long)]
        list: bool,

        /// Sets threshold. Range: [0, 64]
        #[arg(short, long, default_value = "8", value_parser = clap::value_parser!(u8).range(0..=64))]
        threshold: u8,
    },
}

#[derive(Debug)]
struct Image {
    path: PathBuf,
    hash: PerceptualHash,
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, i: usize, j: usize) {
        let mut root_i = self.find(i);
        let mut root_j = self.find(j);

        if root_i == root_j {
            return;
        }

        if self.rank[root_i] < self.rank[root_j] {
            std::mem::swap(&mut root_i, &mut root_j);
        }

        self.parent[root_j] = root_i;

        if self.rank[root_i] == self.rank[root_j] {
            self.rank[root_i] += 1;
        }
    }
}

#[inline]
fn is_image_extension(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref(),
        Some(
            "jpg"
                | "jpeg"
                | "png"
                | "webp"
                | "gif"
                | "bmp"
                | "tiff"
                | "tif"
                | "avif"
                | "ico"
                | "heic"
        )
    )
}

#[inline]
fn load_image(path: &Path) -> Result<DynamicImage, String> {
    let reader = ImageReader::open(path).map_err(|e| format!("{e}"))?;
    let reader = reader.with_guessed_format().map_err(|e| format!("{e}"))?;
    reader.decode().map_err(|e| format!("{e}"))
}

#[inline]
fn group_images(sets: &HashMap<usize, Vec<usize>>, images: &Vec<Image>) {
    let mut i = 0;
    for set in sets {
        if set.1.len() >= 2 {
            i += 1;
            println!("Set {}", i);
            set.1.iter().for_each(|x| println!("{:?}", images[*x].path));
            println!();
        }
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Scan {
            directory,
            recursive,
            list,
            threshold,
        } => {
            let mut failure_count = 0;
            let mut images: Vec<Image> = Vec::new();

            if !directory.exists() {
                eprintln!(
                    "\x1b[31mError:\x1b[0m Directory validation failed\n  Path: {}\n  Reason: Directory does not exist",
                    directory.display()
                );
                std::process::exit(1);
            }

            if !directory.is_dir() {
                eprintln!(
                    "\x1b[31mError:\x1b[0m Directory validation failed\n  Path: {}\n  Reason: The provided path is not a directory",
                    directory.display()
                );
                std::process::exit(1);
            }

            for entries in WalkDir::new(&directory)
                .min_depth(1)
                .max_depth(if recursive { usize::MAX } else { 1 })
            {
                let entry = match entries {
                    Ok(entry) => entry,
                    Err(error) => {
                        eprintln!(
                            "\x1b[31mWarning\x1b[0m: Failed to read directory entry\n  Path: {}\n  Reason: {}",
                            error
                                .path()
                                .map(|path| path.display().to_string())
                                .unwrap_or_else(|| "Unknown".to_string()),
                            error
                        );
                        failure_count += 1;
                        continue;
                    }
                };

                if !entry.file_type().is_file() || !is_image_extension(entry.path()) {
                    continue;
                }

                match load_image(entry.path()) {
                    Ok(image) => {
                        if list {
                            println!(
                                "Image: {:?}\n  Path: {:?}\n  Dimension: {}x{}",
                                entry.file_name(),
                                entry.path(),
                                image.width(),
                                image.height()
                            );
                        }
                        let hash = perceptual_hash(&image);
                        images.push(Image {
                            path: entry.path().to_path_buf(),
                            hash,
                        });
                    }
                    Err(error) => {
                        eprintln!(
                            "\x1b[31mWarning:\x1b[0m Failed to decode an image\n  File: {:?}\n  Reason: {}",
                            entry.path(),
                            error
                        );
                        failure_count += 1;
                    }
                }
            }

            let mut uf = UnionFind::new(images.len());
            for i in 0..images.len() {
                for j in (i + 1)..images.len() {
                    let distance = hamming_distance(images[i].hash, images[j].hash);

                    if distance <= threshold as u32 {
                        uf.union(i, j);
                    }
                }
            }
            let mut component_sizes = vec![0usize; images.len()];
            let mut sets: HashMap<usize, Vec<usize>> = HashMap::new();
            for i in 0..images.len() {
                let root = uf.find(i);
                sets.entry(root).or_default().push(i);
                component_sizes[root] += 1;
            }

            let mut similar_images = 0;
            let mut similar_sets = 0;

            for size in component_sizes {
                if size >= 2 {
                    similar_images += size;
                    similar_sets += 1;
                }
            }

            group_images(&sets, &images);
            println!("Total No. of images: {}", images.len());
            println!("No. of similar images: {}", similar_images);
            println!("No. of sets of similar images: {}", similar_sets);
            println!("Traversal error: {failure_count}");
        }
    }
}
