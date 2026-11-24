"""Build the four-page offline Melody Lab from encoded melodies and experiment exports."""
import argparse
from html import unescape
from pathlib import Path
from shutil import copy2, copytree

from build_data import add_input_arguments, build_data


def build_fonts(output, fonts):
    from fontTools import subset
    from fontTools.ttLib import TTFont

    # Include the text in both catalogues, even when their pages load on demand.
    files = [output / name for name in ("index.html", "app.js", "music.js")]
    files.extend(sorted((output / "data").glob("*.js")))
    characters = {chr(codepoint) for codepoint in range(32, 127)}
    for path in files:
        text = path.read_text(encoding="utf-8")
        characters.update(unescape(text) if path.suffix == ".html" else text)
    codepoints = {ord(character) for character in characters}

    options = subset.Options()
    options.name_IDs = ["*"]
    options.name_languages = ["*"]
    options.name_legacy = True
    font_dir = output / "fonts"
    font_dir.mkdir(exist_ok=True)
    for path, name in fonts:
        with TTFont(path, recalcTimestamp=False) as font:
            glyphs = subset.Subsetter(options=options)
            glyphs.populate(unicodes=codepoints)
            glyphs.subset(font)
            font.flavor = "woff"
            font.save(font_dir / name)


def add_font_faces(stylesheet):
    css = """@font-face {
  font-family: "HYXinHaiXingKaiW";
  src: url("fonts/title.woff") format("woff");
  font-display: swap;
}

@font-face {
  font-family: "STZhongsong";
  src: url("fonts/body.woff") format("woff");
  font-display: swap;
}

"""
    stylesheet.write_text(css + stylesheet.read_text(encoding="utf-8"), encoding="utf-8")


def main():
    source = Path(__file__).resolve().parent
    repo = source.parent
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--output", type=Path, default=repo / "dist" / "melody-lab")
    parser.add_argument("--title-font", type=Path)
    parser.add_argument("--body-font", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("Choose an output directory separate from the website source.")
    if bool(args.title_font) != bool(args.body_font):
        parser.error("--title-font and --body-font must be given together.")
    for option, path in (("--title-font", args.title_font), ("--body-font", args.body_font)):
        if path is not None and not path.is_file():
            parser.error(f"{option} file not found: {path}")
    output.mkdir(parents=True, exist_ok=True)
    build_data(repo / "data", output / "data", args)
    for name in ("index.html", "style.css", "app.js", "music.js", "README.md"):
        copy2(source / name, output / name)
    copytree(source / "vendor", output / "vendor", dirs_exist_ok=True)
    if args.title_font:
        build_fonts(output, [(args.title_font, "title.woff"), (args.body_font, "body.woff")])
        add_font_faces(output / "style.css")
    print(output / "index.html")


if __name__ == "__main__":
    main()
