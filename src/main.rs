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

        /// Lists all the images found.
        #[arg(short, long)]
        show_all_images: bool,

        #[arg(long)]
        show_set: bool,

        /// Sets threshold. Range: [0, 64]
        #[arg(short, long, default_value = "8", value_parser = clap::value_parser!(u8).range(0..=64))]
        threshold: u8,
    },
}

#[inline]
fn valid_directory(directory: &Path) {
    if !directory.exists() {
        eprintln!(
            "\x1b[31mError:\x1b[0m Directory validation failure\n  Path: {:?}\n  Reason: Directory does not exist",
            directory
        );
        std::process::exit(1);
    }

    if !directory.is_dir() {
        eprintln!(
            "\x1b[31mError:\x1b[0m Directory validation failure\n  Path: {:?}\n  Reason: The provided path is not a directory",
            directory
        );
        std::process::exit(1);
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
fn collect_images(images: &mut Vec<Image>, directory: &Path, recursive: bool) {
    for entries in WalkDir::new(directory)
        .min_depth(1)
        .max_depth(if recursive { usize::MAX } else { 1 })
    {
        let entry = match entries {
            Ok(entry) => entry,
            Err(error) => {
                eprintln!(
                    "\x1b[31mWarning\x1b[0m: Failed to read directory entry\n  Path: {:?}\n  Reason: {}",
                    error.path().expect("Unknown"),
                    error
                );
                continue;
            }
        };

        if !entry.file_type().is_file() || !is_image_extension(entry.path()) {
            continue;
        }

        match load_image(entry.path()) {
            Ok(image) => {
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
            }
        }
    }
}

#[inline]
fn form_image_set(
    sets: &mut HashMap<usize, Vec<usize>>,
    images: &[Image],
    uf: &mut UnionFind,
    threshold: u8,
) {
    for i in 0..images.len() {
        for j in (i + 1)..images.len() {
            let distance = hamming_distance(images[i].hash, images[j].hash);

            if distance <= threshold as u32 {
                uf.union(i, j);
            }
        }
    }
    for i in 0..images.len() {
        let root = uf.find(i);
        sets.entry(root).or_default().push(i);
    }
}

#[inline]
fn print_info(
    sets: &HashMap<usize, Vec<usize>>,
    images: &[Image],
    show_set: bool,
    show_all_images: bool,
) {
    if show_all_images {
        images.iter().for_each(|image| println!("{:?}", image.path));
    }

    let mut i: usize = 0;
    for set in sets {
        if set.1.len() >= 2 {
            i += 1;
            if show_set {
                println!("Set {}", i);
                set.1.iter().for_each(|x| println!("{:?}", images[*x].path));
                println!();
            }
        }
    }
    println!("No. of images: {}", images.len());
    println!("No. of image sets: {}", i);
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Scan {
            directory,
            recursive,
            show_all_images,
            threshold,
            show_set,
        } => {
            valid_directory(&directory);
            let mut images: Vec<Image> = Vec::new();
            collect_images(&mut images, &directory, recursive);

            let mut union_find = UnionFind::new(images.len());
            let mut sets: HashMap<usize, Vec<usize>> = HashMap::new();
            form_image_set(&mut sets, &images, &mut union_find, threshold);
            print_info(&sets, &images, show_set, show_all_images);
        }
    }
}
