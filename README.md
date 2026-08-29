# QTGMC Convert

Automatically converts legacy (AviSynth, havsfunc, etc.) QTGMC settings into a [vs-jetpack](https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack) compatible preset.

Temporarily requires git latest of vs-jetpack until a new stable release is made.

## Note on usage
This tool is intended solely for matching legacy AviSynth/havsfunc behavior in vs-jetpack.
vs-jetpack's native defaults provide significantly higher visual quality and faster performance.
Using this converter is not recommended unless strict preset parity with havsfunc is desired.
If you're either a new user or prioritize image quality, you probably shouldn't use this.

The output of this settings mapping will not (and cannot be) bit-exact.
vs-jetpack uses different plugins, comprehensive algorithm optimizations, and numerous bug fixes.
Reduced intermediate rounding also improves overall processing precision at any bit depth.
Some algorithms (notably sharpening) have been reworked; while visual sharpness matches,
the output is not identical. Output variations are more noticeable at higher bit depths
due to broken scaling in legacy implementations.

## How do I use this?

All you have to do is replace your `QTGMC()` call with `qtgmc_convert()`, like so:

```py
from havsfunc import QTGMC
from vsdeinterlace import QTempGaussMC
from qtgmc_convert import qtgmc_convert

# havsfunc call example.
old = QTGMC(clip, Preset="Very Slow", NoisePreset="Medium")

# Use the same settings in the preset converter.
settings = qtgmc_convert(Preset="Very Slow", NoisePreset="Medium")

# Unpack the converted settings.
qtgmc = QTempGaussMC(**settings)

# Deinterlace.
deinterlaced = qtgmc.deinterlace(clip)
```
