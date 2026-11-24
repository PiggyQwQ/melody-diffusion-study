"""Build a Windows x64 desktop package from the offline Melody Lab website."""
import argparse
from hashlib import sha256
from os import environ
from pathlib import Path
from shutil import copy2, copyfileobj
import subprocess
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile, ZIP_DEFLATED


SDK_VERSION = "1.0.4370.26"
SDK_NAME = f"microsoft.web.webview2.{SDK_VERSION}.nupkg"
SDK_URL = f"https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/{SDK_VERSION}/{SDK_NAME}"
SDK_SHA256 = "1c0dacac84b70109b3fd42f90c53a0ba942abc3a87590d7a4e0ccd86e4a06cf4"
SDK_FILES = {
    "lib/net462/Microsoft.Web.WebView2.Core.dll": "Microsoft.Web.WebView2.Core.dll",
    "lib/net462/Microsoft.Web.WebView2.WinForms.dll": "Microsoft.Web.WebView2.WinForms.dll",
    "runtimes/win-x64/native/WebView2Loader.dll": "WebView2Loader.dll",
    "LICENSE.txt": "WebView2-LICENSE.txt",
    "NOTICE.txt": "WebView2-NOTICE.txt",
}
SITE_FILES = (
    "index.html", "style.css", "app.js", "music.js", "README.md",
    "data/experiments.js", "data/library-folk.js", "data/library-pop.js",
    "vendor/vexflow.js", "vendor/VexFlow-LICENSE.txt", "vendor/VexFlow-Font-LICENSES.txt",
)
OPTIONAL_FONTS = ("fonts/title.woff", "fonts/body.woff")


def get_sdk(cache):
    cache.mkdir(parents=True, exist_ok=True)
    package = cache / SDK_NAME
    if not package.exists():
        with TemporaryDirectory(dir=cache) as temp:
            downloaded = Path(temp) / SDK_NAME
            with urlopen(SDK_URL, timeout=35) as response, downloaded.open("wb") as target:
                copyfileobj(response, target)
            if sha256(downloaded.read_bytes()).hexdigest() != SDK_SHA256:
                raise SystemExit("The downloaded WebView2 SDK checksum does not match.")
            downloaded.replace(package)
    if sha256(package.read_bytes()).hexdigest() != SDK_SHA256:
        raise SystemExit(f"WebView2 SDK checksum does not match: {package}")
    return package


def build(package, archive, source, site, files, compiler, sdk, name):
    package.mkdir()
    with ZipFile(sdk) as dependency:
        for member, filename in SDK_FILES.items():
            (package / filename).write_bytes(dependency.read(member))

    for filename in files:
        destination = package / "site" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(site / filename, destination)
    copy2(source / "使用说明.txt", package / "使用说明.txt")

    subprocess.run([
        str(compiler), "/nologo", "/target:winexe", "/platform:x64", "/optimize+",
        "/utf8output", "/codepage:65001", f"/out:{package / 'MelodyLab.exe'}",
        f"/win32manifest:{source / 'app.manifest'}",
        "/reference:System.Windows.Forms.dll",
        "/reference:System.Drawing.dll",
        f"/reference:{package / 'Microsoft.Web.WebView2.Core.dll'}",
        f"/reference:{package / 'Microsoft.Web.WebView2.WinForms.dll'}",
        str(source / "Program.cs"),
    ], check=True)

    with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                bundle.write(path, f"{name}/{path.relative_to(package).as_posix()}")


def main():
    source = Path(__file__).resolve().parent
    dist = source.parents[1] / "dist"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=dist / "melody-lab")
    parser.add_argument("--output", type=Path, default=dist / "melody-lab-windows-x64")
    parser.add_argument("--cache", type=Path, default=dist / "webview2-sdk")
    args = parser.parse_args()
    site, output = args.site.resolve(), args.output.resolve()
    if output == site or output.is_relative_to(site) or site.is_relative_to(output):
        parser.error("The desktop output and website directories must be separate.")
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("Choose a desktop output directory separate from the source.")
    missing = [name for name in SITE_FILES if not (site / name).is_file()]
    if missing:
        parser.error(f"The website at {site} is missing: {', '.join(missing)}")
    css = (site / "style.css").read_text(encoding="utf-8")
    fonts = tuple(name for name in OPTIONAL_FONTS if name in css)
    missing_fonts = [name for name in fonts if not (site / name).is_file()]
    if missing_fonts:
        parser.error(f"The website at {site} is missing fonts: {', '.join(missing_fonts)}")

    compiler = (Path(environ.get("WINDIR", "C:/Windows")) / "Microsoft.NET" /
                "Framework64" / "v4.0.30319" / "csc.exe")
    if not compiler.is_file():
        parser.error("The Windows x64 .NET Framework C# compiler was not found.")
    sdk = get_sdk(args.cache.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = output.with_suffix(".zip")
    with TemporaryDirectory(prefix=f"{output.name}-build-", dir=output.parent) as temp:
        package, staged_zip = Path(temp) / "app", Path(temp) / archive.name
        build(package, staged_zip, source, site, SITE_FILES + fonts, compiler, sdk, output.name)
        for path in sorted(package.rglob("*")):
            if path.is_file():
                destination = output / path.relative_to(package)
                destination.parent.mkdir(parents=True, exist_ok=True)
                copy2(path, destination)
        staged_zip.replace(archive)
    print(output / "MelodyLab.exe")
    print(f"{archive}: {archive.stat().st_size:,} bytes, SHA-256 {sha256(archive.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
