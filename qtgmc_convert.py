import warnings
from collections.abc import Mapping
from functools import partial
from math import sqrt
from types import MappingProxyType, SimpleNamespace
from typing import Any

from jetpytools import CustomEnum, KwargsNotNone, fallback
from vsaa import BWDIF, EEDI3, NNEDI3
from vsdeinterlace import QTempGaussMC
from vsdenoise import AnalyzeArgs, DFTTest, MVToolsPreset, RecalculateArgs, SuperArgs, bm3d, nl_means
from vskernels import Bobber, Catrom
from vstools import vs

_SLMode = {
    0: QTempGaussMC.SharpenLimitMode.NONE,
    1: QTempGaussMC.SharpenLimitMode.SPATIAL_PRESMOOTH,
    2: QTempGaussMC.SharpenLimitMode.TEMPORAL_PRESMOOTH,
    3: QTempGaussMC.SharpenLimitMode.SPATIAL_POSTSMOOTH,
    4: QTempGaussMC.SharpenLimitMode.TEMPORAL_POSTSMOOTH,
}

_Sbb = {
    0: QTempGaussMC.BackBlendMode.NONE,
    1: QTempGaussMC.BackBlendMode.PRELIMIT,
    2: QTempGaussMC.BackBlendMode.POSTLIMIT,
    3: QTempGaussMC.BackBlendMode.BOTH,
}

_SrchClipPP = {
    0: ((0, 0), (0, 0, 0)),
    1: ((sqrt(2.75), 1), (0, 0, 0)),
    2: ((sqrt(3.5625), 0.9), (0, 0, 0)),
    3: ((sqrt(3.5625), 0.9), (3, 7, 2)),
}

_Lossless = {
    0: QTempGaussMC.LosslessMode.NONE,
    1: QTempGaussMC.LosslessMode.POSTSMOOTH,
    2: QTempGaussMC.LosslessMode.PRESHARPEN,
}

_NoiseDeint = {
    "DOUBLEWEAVE": QTempGaussMC.NoiseDeintMode.WEAVE,
    "BOB": QTempGaussMC.NoiseDeintMode.BOB,
    "GENERATE": QTempGaussMC.NoiseDeintMode.GENERATE,
}


class _Preset(CustomEnum):
    DRAFT = MappingProxyType(
        {
            "TR0": 0,
            "TR1": 1,
            "TR2": 0,
            "Rep0": 0,
            "Rep2": 0,
            "EdiMode": "BOB",
            "NNSize": 4,
            "NNeurons": 0,
            "EdiMaxD": 4,
            "SMode": 0,
            "SLMode": 0,
            "SLRad": 1,
            "Sbb": 0,
            "SrchClipPP": 0,
            "SubPel": 1,
            "BlockSize": 32,
            "Overlap": 32 // 4,
            "Search": 0,
            "SearchParam": 1,
            "PelSearch": 1,
            "ChromaMotion": False,
            "Precise": False,
            "ProgSADMask": 0,
        }
    )

    ULTRA_FAST = MappingProxyType(dict(DRAFT) | {"TR0": 1, "Rep2": 3, "EdiMode": "BWDIF", "SMode": 2, "SrchClipPP": 1})

    SUPER_FAST = MappingProxyType(dict(ULTRA_FAST) | {"EdiMode": "NNEDI3"})

    VERY_FAST = MappingProxyType(
        dict(SUPER_FAST) | {"Rep2": 4, "EdiMaxD": 5, "SLMode": 2, "SrchClipPP": 2, "Search": 4}
    )

    FASTER = MappingProxyType(dict(VERY_FAST) | {"EdiMaxD": 6, "Overlap": 32 // 2, "SearchParam": 2})

    FAST = MappingProxyType(dict(FASTER) | {"TR0": 2, "Rep0": 3, "NNSize": 5, "BlockSize": 16, "Overlap": 16 // 2})

    MEDIUM = MappingProxyType(dict(FAST) | {"TR2": 1, "NNeurons": 1, "EdiMaxD": 7, "SrchClipPP": 3, "ProgSADMask": 10})

    SLOW = MappingProxyType(dict(MEDIUM) | {"Rep0": 4, "NNSize": 1, "SubPel": 2, "PelSearch": 2})

    SLOWER = MappingProxyType(dict(SLOW) | {"TR1": 2, "EdiMaxD": 8, "Sbb": 1, "ChromaMotion": True})

    VERY_SLOW = MappingProxyType(dict(SLOWER) | {"TR2": 2, "NNeurons": 2, "EdiMaxD": 10, "Precise": True})

    PLACEBO = MappingProxyType(dict(VERY_SLOW) | {"TR2": 3, "EdiMaxD": 12, "SLRad": 3, "Sbb": 3, "Search": 5})


class _NoisePreset(CustomEnum):
    FASTER = MappingProxyType(
        {"Denoiser": "FFT3D", "DenoiseMC": False, "NoiseTR": 0, "NoiseDeint": "DOUBLEWEAVE", "StabilizeNoise": False}
    )

    FAST = MappingProxyType(dict(FASTER) | {"NoiseTR": 1})

    MEDIUM = MappingProxyType(dict(FAST) | {"Denoiser": "DFTTEST", "StabilizeNoise": True})

    SLOW = MappingProxyType(dict(MEDIUM) | {"DenoiseMC": True, "NoiseDeint": "BOB"})

    SLOWER = MappingProxyType(dict(SLOW) | {"NoiseTR": 2, "NoiseDeint": "GENERATE"})


class _MatchPreset2(CustomEnum):
    ULTRA_FAST = MappingProxyType({"MatchEdi2": "BOB", "MatchNNSize2": 4, "MatchNNeurons2": 0, "MatchEdiMaxD2": 4})

    SUPER_FAST = MappingProxyType(dict(ULTRA_FAST) | {"MatchEdi2": "NNEDI3"})

    VERY_FAST = MappingProxyType(dict(SUPER_FAST) | {"MatchEdiMaxD2": 5})

    FASTER = MappingProxyType(dict(VERY_FAST) | {"MatchEdiMaxD2": 6})

    FAST = MappingProxyType(dict(FASTER) | {"MatchNNSize2": 5})

    MEDIUM = MappingProxyType(dict(FAST) | {"MatchNNeurons2": 1, "MatchEdiMaxD2": 7})

    SLOW = MappingProxyType(dict(MEDIUM) | {"MatchNNSize2": 1})

    SLOWER = MappingProxyType(dict(SLOW) | {"MatchEdiMaxD2": 8})

    VERY_SLOW = MappingProxyType(dict(SLOWER) | {"MatchNNeurons2": 2, "MatchEdiMaxD2": 10})

    PLACEBO = MappingProxyType(dict(VERY_SLOW) | {"MatchEdiMaxD2": 12})


def qtgmc_convert(
    Input: vs.VideoNode | None = None,
    Preset: str = "Slower",
    TR0: int | None = None,
    TR1: int | None = None,
    TR2: int | None = None,
    Rep0: int | None = None,
    Rep1: int = 0,
    Rep2: int | None = None,
    EdiMode: str | None = None,
    RepChroma: bool = True,
    NNSize: int | None = None,
    NNeurons: int | None = None,
    EdiQual: int = 1,
    EdiMaxD: int | None = None,
    ChromaEdi: str = "",
    EdiExt: vs.VideoNode | None = None,
    Sharpness: float | None = None,
    SMode: int | None = None,
    SLMode: int | None = None,
    SLRad: int | None = None,
    SOvs: int = 0,
    SVThin: float = 0.0,
    Sbb: int | None = None,
    SrchClipPP: int | None = None,
    SubPel: int | None = None,
    SubPelInterp: int = 2,
    BlockSize: int | None = None,
    Overlap: int | None = None,
    Search: int | None = None,
    SearchParam: int | None = None,
    PelSearch: int | None = None,
    ChromaMotion: bool | None = None,
    TrueMotion: bool = False,
    Lambda: int | None = None,
    LSAD: int | None = None,
    PNew: int | None = None,
    PLevel: int | None = None,
    GlobalMotion: bool = True,
    DCT: int = 0,
    ThSAD1: int = 640,
    ThSAD2: int = 256,
    ThSCD1: int = 180,
    ThSCD2: int = 98,
    SourceMatch: int = 0,
    MatchPreset: str | None = None,
    MatchEdi: str | None = None,
    MatchPreset2: str | None = None,
    MatchEdi2: str | None = None,
    MatchTR2: int = 1,
    MatchEnhance: float = 0.5,
    Lossless: int = 0,
    NoiseProcess: int | None = None,
    EZDenoise: float | None = None,
    EZKeepGrain: float | None = None,
    NoisePreset: str = "Fast",
    Denoiser: str | None = None,
    FftThreads: int = 1,
    DenoiseMC: bool | None = None,
    NoiseTR: int | None = None,
    Sigma: float | None = None,
    ChromaNoise: bool = False,
    ShowNoise: bool | float = 0.0,
    GrainRestore: float | None = None,
    NoiseRestore: float | None = None,
    NoiseDeint: str | None = None,
    StabilizeNoise: bool | None = None,
    InputType: int = 0,
    ProgSADMask: float | None = None,
    FPSDivisor: int = 1,
    ShutterBlur: int = 0,
    ShutterAngleSrc: float = 180.0,
    ShutterAngleOut: float = 180.0,
    SBlurLimit: int = 4,
    Border: bool = False,
    Precise: bool | None = None,
    Tuning: str = "None",
    ShowSettings: bool = False,
    GlobalNames: str = "QTGMC",
    PrevGlobals: str = "Replace",
    ForceTR: int = 0,
    Str: float = 2.0,
    Amp: float = 0.0625,
    FastMA: bool = False,
    ESearchP: bool = False,
    RefineMotion: bool = False,
    TFF: bool | None = None,
    nnedi3_args: Mapping[str, Any] = {},
    eedi3_args: Mapping[str, Any] = {},
    opencl: bool = False,
    device: int | None = None,
) -> dict[str, Any]:
    """
    QTGMC Convert

    Automatically converts AviSynth/havsfunc QTGMC settings into an equivalent vs-jetpack preset.

    WARNING:
        This tool is intended solely for matching legacy AviSynth/havsfunc behavior in vs-jetpack.
        vs-jetpack's native defaults provide significantly higher visual quality and faster performance.
        Using this converter is not recommended unless strict preset parity with havsfunc is desired.

    Note:
        The output of this settings mapping will not (and cannot be) bit-exact.
        vs-jetpack uses different plugins, comprehensive algorithm optimizations, and numerous bug fixes.
        Reduced intermediate rounding also improves overall processing precision at any bit depth.
        Some algorithms (notably sharpening) have been reworked; while visual sharpness matches,
        the output is not identical. Output variations are more noticeable at higher bit depths
        due to broken scaling in AviSynth/havsfunc.

    Usage example:
        ```python
        # havsfunc call example.
        old = QTGMC(Preset="Very Slow", NoisePreset="Medium")

        # Use the same settings in the preset converter.
        settings = qtgmc_convert(Preset="Very Slow", NoisePreset="Medium")

        # Unpack the converted settings for vs-jetpack.
        qtgmc = QTempGaussMC(**settings)

        deinterlaced = qtgmc.deinterlace(clip)
        ```

    Returns:
        A settings preset dictionary compatible with `vsjetpack.QTempGaussMC`.
    """

    def map_rep(Rep: int) -> dict[str, int]:
        ed = Rep if Rep < 10 else Rep // 10
        od = 0 if Rep < 10 else Rep % 10

        return {"erosion_distance": ed, "over_dilation": od}

    def map_edi(edimode: str, nsize: int, nns: int, qual: int, mdis: int) -> Bobber:
        edimode_nnedi3_args = {"nsize": nsize, "nns": nns, "qual": qual, "pscrn": 2, "gpu": opencl} | nnedi3_args
        edimode_eedi3_args = {
            "mdis": mdis,
            "backend": EEDI3.Backend.OPENCL if opencl else EEDI3.Backend.CPU,
        } | eedi3_args

        match edimode.upper():
            case "NNEDI3":
                edimode = NNEDI3(**edimode_nnedi3_args)
            case "EEDI3+NNEDI3":
                edimode = EEDI3(sclip=NNEDI3(**edimode_nnedi3_args), **edimode_eedi3_args)
            case "EEDI3":
                edimode = EEDI3(**edimode_eedi3_args)
            case "BWDIF":
                edimode = BWDIF()
            case _:
                edimode = Catrom()

        return edimode

    Preset = _Preset[Preset.replace(" ", "_").upper()]
    NoisePreset = _NoisePreset[NoisePreset.replace(" ", "_").upper()]

    if MatchPreset:
        MatchPreset = _MatchPreset2[MatchPreset.replace(" ", "_").upper()]
    else:
        MatchPreset = tuple(_MatchPreset2)[max(tuple(_Preset).index(Preset) - 4, 0)]

    if MatchPreset2:
        MatchPreset2 = _MatchPreset2[MatchPreset2.replace(" ", "_").upper()]
    else:
        MatchPreset2 = tuple(_MatchPreset2)[max(tuple(_Preset).index(Preset) - 6, 0)]

    if bool((MatchPreset or MatchEdi) and SourceMatch):
        warnings.warn(
            "Basic source match interpolation always matches the main interpolation in vs-jetpack. "
            f"MatchPreset ({MatchPreset.name}) and MatchEdi have been dropped.",
            UserWarning,
        )

    psv = SimpleNamespace(
        **Preset.value
        | NoisePreset.value
        | MatchPreset2.value
        | KwargsNotNone(
            TR0=TR0,
            TR1=TR1,
            Rep0=Rep0,
            Rep2=Rep2,
            EdiMode=EdiMode,
            NNSize=NNSize,
            NNeurons=NNeurons,
            EdiMaxD=EdiMaxD,
            SMode=SMode,
            SLRad=SLRad,
            Sbb=Sbb,
            SrchClipPP=SrchClipPP,
            SubPel=SubPel,
            Overlap=Overlap,
            SearchParam=SearchParam,
            PelSearch=PelSearch,
            ChromaMotion=ChromaMotion,
            Precise=Precise,
            ProgSADMask=ProgSADMask,
            Denoiser=Denoiser,
            DenoiseMC=DenoiseMC,
            NoiseTR=NoiseTR,
            NoiseDeint=NoiseDeint,
            StabilizeNoise=StabilizeNoise,
            MatchEdi2=MatchEdi2,
        )
    )

    psv.TR2 = fallback(TR2, max(psv.TR2, bool(SourceMatch)))
    psv.SLMode = fallback(SLMode, 0 if SourceMatch else psv.SLMode)
    psv.BlockSize = fallback(BlockSize, 32 if Tuning.upper() == "DV-HD" else psv.BlockSize)
    psv.Search = fallback(Search, psv.Search) - 2  # Shift down for MVUtensils.

    if ESearchP and psv.Search in (2, 3):
        if Preset in (_Preset.PLACEBO, _Preset.VERY_SLOW, _Preset.SLOWER, _Preset.SLOW):
            psv.SearchParam = 24
        else:
            psv.SearchParam = 16

    Lambda = fallback(Lambda, 1000 if TrueMotion else 100)
    LSAD = fallback(LSAD, 1200 if TrueMotion else 400)
    PNew = fallback(PNew, 50 if TrueMotion else 25)
    PLevel = fallback(PLevel, 1 if TrueMotion else 0)

    if psv.Search < 0:
        psv.Search = 0
        warnings.warn("Search mode unsupported by MVUtensils. Value has been remapped to diamond.", UserWarning)

    match DCT:
        case 0:
            DCT = False
        case 5:
            DCT = True
        case _:
            DCT = True
            warnings.warn("DCT mode unsupported by MVUtensils. Value has been remapped to SATD.", UserWarning)

    analyze_preset = MVToolsPreset(
        pel=psv.SubPel,
        pad=psv.BlockSize,
        chroma=psv.ChromaMotion,
        super_args=SuperArgs(sharp=SubPelInterp),
        analyze_args=AnalyzeArgs(
            search=psv.Search,
            searchparam=psv.SearchParam,
            pelsearch=psv.PelSearch,
            mvlambda=Lambda,
            lsad=LSAD,
            plevel=PLevel,
            globalmv=GlobalMotion,
            pnew=PNew,
            satd=DCT,
        ),
        recalculate_args=RecalculateArgs(
            search=psv.Search,
            searchparam=psv.SearchParam,
            pelsearch=psv.PelSearch,
            mvlambda=Lambda,
            lsad=LSAD,
            plevel=PLevel,
            globalmv=GlobalMotion,
            pnew=PNew,
            satd=DCT,
        ),
    )

    Sharpness = fallback(Sharpness, 0 if not psv.SMode else 0.2 if SourceMatch else 1) * (
        (2 if psv.SLMode in (2, 4) else 1.5 if psv.SLMode in (1, 3) else 1) * (0.2 + psv.TR1 * 0.15 + psv.TR2 * 0.25)
        + (0.1 if psv.SMode == 1 else 0)
    )

    NoiseProcess = fallback(
        NoiseProcess, 1 if EZDenoise else 2 if EZKeepGrain or Preset in (_Preset.PLACEBO, _Preset.VERY_SLOW) else 0
    )
    GrainRestore = (
        fallback(
            GrainRestore, 0 if EZDenoise else 0.3 * sqrt(EZKeepGrain) if EZKeepGrain else [0, 0.7, 0.3][NoiseProcess]
        )
        if NoiseProcess
        else 0
    )
    NoiseRestore = (
        fallback(
            NoiseRestore, 0 if EZDenoise else 0.1 * sqrt(EZKeepGrain) if EZKeepGrain else [0, 0.3, 0.1][NoiseProcess]
        )
        if NoiseProcess
        else 0
    )
    Sigma = fallback(Sigma, EZDenoise if EZDenoise else 4 * EZKeepGrain if EZKeepGrain else 2)

    prefilter_strength, prefilter_limit = _SrchClipPP[psv.SrchClipPP]

    match psv.Denoiser.upper():
        case "BM3D":
            psv.Denoiser = partial(bm3d, sigma=Sigma)
        case "DFTTEST":
            psv.Denoiser = DFTTest(sigma=Sigma * 4)
        case "KNLMEANSCL":
            psv.Denoiser = partial(nl_means, h=Sigma)
        case _:
            psv.Denoiser = DFTTest(sigma=Sigma * 4)
            warnings.warn(
                "FFT3D is unsupported in modern versions of VapourSynth, value has been remapped to DFTTest.",
                UserWarning,
            )

    # Start raising errors/warns for unsupported settings.
    unsupported_settings = {
        "RepChroma": not RepChroma,
        "ChromaEdi": ChromaEdi,
        "ChromaNoise": not ChromaNoise,
        "ShutterBlur": ShutterBlur > 1,
        "FastMA": FastMA,
        "device": device is not None,
    }
    for name, value in unsupported_settings.items():
        if value:
            warnings.warn(f"{name} is unsupported in vs-jetpack and has been dropped.", UserWarning)

    manual_settings = {
        "EdiExt": EdiExt,
        "ShowNoise": ShowNoise,
        "InputType": InputType is not None,
        "ShowSettings": ShowSettings,
        "TFF": TFF is not None,
        "Border": Border,
        "PrevGlobals": PrevGlobals.upper() == "REUSE",
    }
    for name, value in manual_settings.items():
        if value:
            warnings.warn(
                f"{name} cannot be automatically converted. See manual migration guide for details.", UserWarning
            )

    return {
        "prefilter__tr": psv.TR0,
        "basic__tr": psv.TR1,
        "final__tr": psv.TR2,
        "prefilter__mask_shimmer_args": map_rep(psv.Rep0),
        "basic__mask_shimmer_args": map_rep(Rep1),
        "final__mask_shimmer_args": map_rep(psv.Rep2),
        "basic__bobber": map_edi(psv.EdiMode, psv.NNSize, psv.NNeurons, EdiQual, psv.EdiMaxD),
        # Scale is Sharpness * (binomial_variance / gaussian_variance);
        # where binomial_variance is the blur used in the original unsharpening
        # and gaussian_variance is the blur used in vs-jetpack's unsharpening.
        "sharpen__strength": Sharpness * (0.5 / var) if (var := psv.TR1 / 2 + psv.TR2 * (psv.TR2 + 1) / 3) else 0,
        "sharpen__offset": (1 if psv.Precise else 0) if psv.SMode == 2 else False,
        "sharpen_limit__mode": _SLMode[psv.SLMode],
        "sharpen_limit__radius": psv.SLRad,
        # Disable SOvs for spatial limiting, AVS/havsfunc don't support it.
        "sharpen_limit__clamp": SOvs if psv.SLMode in (2, 4) else 0,
        # Rescale SVThin to match in vs-jetpack.
        "sharpen__thin": SVThin * 6 / (32 / sqrt(105)),
        "back_blend__mode": _Sbb[psv.Sbb],
        "prefilter__strength": prefilter_strength,
        "prefilter__limit": prefilter_limit,
        "analyze__preset": analyze_preset,
        "analyze__blksize": psv.BlockSize,
        "analyze__overlap": psv.BlockSize // psv.Overlap if psv.Overlap else 0,
        "basic__thsad": ThSAD1,
        "final__thsad": ThSAD2,
        "analyze__thscd": (ThSCD1, ThSCD2 * 100 / 256),  # Rescale for MVUtensils.
        "source_match__iterations": SourceMatch,
        "source_match__bobber": map_edi(psv.MatchEdi2, psv.MatchNNSize2, psv.MatchNNeurons2, 1, psv.MatchEdiMaxD2),
        "source_match__tr": MatchTR2,
        "source_match__enhance": MatchEnhance,
        "lossless__mode": _Lossless[Lossless],
        "denoise__full_denoise": NoiseProcess == 1,
        "denoise__func": psv.Denoiser,
        "denoise__mc_denoise": psv.DenoiseMC,
        "denoise__tr": psv.NoiseTR,
        "basic__noise_restore": GrainRestore,
        "final__noise_restore": NoiseRestore,
        "denoise__deint": _NoiseDeint.get(psv.NoiseDeint.upper(), QTempGaussMC.NoiseDeintMode.WEAVE),
        "denoise__stabilize": 0.4 if psv.StabilizeNoise else False,
        "basic__mask_args": {"ml": psv.ProgSADMask},
        "motion_blur__fps_divisor": FPSDivisor,
        "motion_blur__shutter_angle": (ShutterAngleSrc, ShutterAngleOut) if ShutterBlur else False,
        "motion_blur__mask_args": {"ml": SBlurLimit},
        "analyze__force_tr": ForceTR,
        "prefilter__range_expansion_args": {"slope": Str, "smooth": Amp},
        "analyze__refine": int(RefineMotion),
        # Set defaults for some hardcoded settings.
        "prefilter__sc_threshold": 28 / 255,
        "analyze__thsad_recalc": ThSAD1 // 2,
    }
