// This file is part of text-to-audiobook
//
// text-to-audiobook is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// text-to-audiobook is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! ID3 metadata for chunk WAVs, stored in the RIFF `ID3 ` chunk.

use std::path::Path;

use chrono::Datelike;
use id3::frame::Comment;
use id3::{Tag, TagLike, Version};

use crate::segment::Segment;

/// Existing tag of a WAV file, if any.
pub fn read_tag(path: &Path) -> Option<Tag> {
    Tag::read_from_path(path).ok()
}

/// Write `tag` into the WAV file (failures are logged).
pub fn write_tag(path: &Path, tag: &Tag) {
    if let Err(e) = tag.write_to_path(path, Version::Id3v23) {
        log::warn!("Failed to tag WAV file {}: {e}", path.display());
    }
}

/// Tag a chunk WAV with title, album (story), track (chapter), artist,
/// genre, year and a comment, like the Python version:
/// - TIT2 "{chapter_title} - Part {chunk_number}"
/// - TALB story title, TRCK chapter number, TPE1 "Narrator"
/// - TCON "Audiobook", TYER current year
/// - COMM "Chunk {chunk_number}, Chapter {chapter_number}"
pub fn tag_wav(path: &Path, seg: &Segment) {
    let chunk_number = seg.chunk_number;
    let chapter_number = seg.chapter_number;
    let chapter_title = seg
        .chapter_title
        .clone()
        .unwrap_or_else(|| format!("Chapter {chapter_number}"));
    let story_title = seg
        .story_title
        .clone()
        .unwrap_or_else(|| "Audiobook".into());

    let mut tag = read_tag(path).unwrap_or_default();
    tag.set_title(format!("{chapter_title} - Part {chunk_number}"));
    tag.set_album(story_title);
    tag.set_text("TRCK", chapter_number.to_string());
    tag.set_artist("Narrator");
    tag.set_genre("Audiobook");
    tag.set_year(chrono::Local::now().year());
    tag.remove("COMM");
    tag.add_frame(Comment {
        lang: "eng".into(),
        description: String::new(),
        text: format!("Chunk {chunk_number}, Chapter {chapter_number}"),
    });
    write_tag(path, &tag);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_survive_and_audio_stays_readable() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("chunk_000003.wav");
        crate::audio::write_mono_wav(&p, &[0.1; 480], 24000).unwrap();
        let seg = Segment {
            chunk_number: 3,
            chapter_number: 2,
            chapter_title: Some("Der Anfang".into()),
            story_title: Some("Mein Buch".into()),
            ..Segment::with_text("x")
        };
        tag_wav(&p, &seg);
        tag_wav(&p, &seg);
        let t = read_tag(&p).unwrap();
        assert_eq!(t.title(), Some("Der Anfang - Part 3"));
        assert_eq!(t.album(), Some("Mein Buch"));
        assert_eq!(t.comments().count(), 1);
        assert_eq!(crate::audio::read_wav(&p).unwrap().samples.len(), 480);
        crate::audio::filter::denoise(&p).unwrap();
        assert_eq!(read_tag(&p).unwrap().album(), Some("Mein Buch"));
    }
}
