use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};

#[allow(dead_code)]
pub struct RemoteFile {
    url: String,
    response: reqwest::blocking::Response,
    total_size: usize,
    current_size: usize,
    pbar: Option<ProgressBar>,
}
impl RemoteFile {
    pub fn with_pbar(url: &str) -> Result<Self> {
        Self::with_config(url, 3600, true)
    }
    pub fn with_config(url: &str, timeout: u64, pbar: bool) -> Result<Self> {
        let client = reqwest::blocking::Client::new();
        let response = client
            .get(url)
            .timeout(std::time::Duration::from_secs(timeout))
            .send()?;
        let total_size = response.content_length().unwrap(); // FIXME

        let pbar = if pbar {
            let pbar = ProgressBar::new(total_size);
            pbar.set_style(ProgressStyle::default_bar()
                .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
                .progress_chars("#>-"));
            pbar.set_message(format!("Downloading {}", url));
            Some(pbar)
        } else {
            None
        };
        Ok(Self {
            url: url.to_owned(),
            response,
            current_size: 0,
            total_size: total_size as usize,
            pbar,
        })
    }
    fn update(&mut self, size: usize) {
        self.current_size += size;
        if let Some(pbar) = &self.pbar {
            pbar.set_position(self.current_size as u64);
        }
    }
}

impl std::io::Read for RemoteFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let size = self.response.read(buf)?;
        self.update(size);
        Ok(size)
    }
}
