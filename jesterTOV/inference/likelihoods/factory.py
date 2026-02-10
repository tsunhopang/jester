r"""Factory functions for creating likelihoods from configuration"""

from pathlib import Path

from ..config.schema import LikelihoodConfig
from .combined import CombinedLikelihood, ZeroLikelihood
from .gw import GWLikelihood, GWLikelihoodResampled
from .nicer import NICERLikelihood
from .radio import RadioTimingLikelihood
from .chieft import ChiEFTLikelihood
from .constraints import (
    ConstraintEOSLikelihood,
    ConstraintTOVLikelihood,
    ConstraintGammaLikelihood,
)
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")

# Preset flow model directories for GW events and NICER PSR with trained flows
# Paths are relative to jesterTOV/inference/ directory
GW_EVENT_PRESETS = {
    "GW170817": "flows/models/gw_maf/gw170817/gw170817_xp_nrtv3",
    "GW190425": "flows/models/gw_maf/gw190425/gw190425_xp_nrtv3",
}

# TODO: update the nicer models together with newer data
NICER_EVENT_PRESETS = {
    "J0030+0451": {
        "Amsterdam": "flows/models/nicer_maf/J00300451/J00300451_amsterdam_ST_PST_NICER_only_Riley2019",
        "Maryland": "flows/models/nicer_maf/J00300451/J00300451_maryland_3spot_NICER_only_full",
    },
    "J0740+6620": {
        "Amsterdam": "flows/models/nicer_maf/J07406620/J07406620_amsterdam_gamma_NICERXMM_equal_weights_recent",
        "Maryland": "flows/models/nicer_maf/J07406620/J07406620_maryland_unknown_NICERXMM_RM",
    },
}

# Aliases for NICER pulsars (both formats supported)
NICER_ALIASES = {
    "J00300451": "J0030+0451",
    "J0030": "J0030+0451",
    "J07406620": "J0740+6620",
    "J0740": "J0740+6620",
}


def get_model_dir(
    event_name: str, model_dir: str | None = None, model_grp: str | None = None
) -> str:
    """
    Get model directory for GW event or NICER pulsar, using presets if path is not provided.

    Parameters
    ----------
    event_name : str
        Name of the event. For GW: 'GW170817', 'GW190425'.
        For NICER: 'J0030+0451', 'J0740+6620' (or aliases like 'J00300451', 'J0030')
    model_dir : str | None, optional
        User-provided model directory. If None/empty, uses preset.
    model_grp : str | None, optional
        For NICER pulsars only: 'Amsterdam' or 'Maryland'.
        Required if model_dir is not provided and event is a NICER pulsar.

    Returns
    -------
    str
        Absolute path to model directory

    Raises
    ------
    ValueError
        If model_dir is not provided and event is not in presets, or if
        model_grp is required but not provided for NICER pulsars.

    Examples
    --------
    >>> # GW event with preset
    >>> get_model_dir("GW170817")

    >>> # NICER pulsar with preset
    >>> get_model_dir("J0030+0451", model_grp="Amsterdam")
    >>> get_model_dir("J00300451", model_grp="Maryland")  # Using alias

    >>> # Custom model directory
    >>> get_model_dir("GW170817", model_dir="/path/to/custom/model")
    """
    # If model_dir is provided and not empty, use it directly
    if model_dir:
        return str(Path(model_dir).resolve())

    # Normalize event name for lookup
    event_name_upper = event_name.upper()

    # Check if it's a GW event
    if event_name_upper in GW_EVENT_PRESETS:
        preset_path = GW_EVENT_PRESETS[event_name_upper]
        inference_dir = Path(__file__).parent.parent
        model_dir_abs = (inference_dir / preset_path).resolve()

        logger.warning(
            f"No model_dir provided for GW event '{event_name}'. "
            f"Using default preset path: {model_dir_abs}"
        )
        return str(model_dir_abs)

    # Check if it's a NICER pulsar (handle aliases)
    nicer_canonical = NICER_ALIASES.get(event_name_upper, event_name_upper)

    if nicer_canonical in NICER_EVENT_PRESETS:
        # NICER pulsars require model_grp
        if not model_grp:
            available_types = list(NICER_EVENT_PRESETS[nicer_canonical].keys())
            raise ValueError(
                f"For NICER pulsar '{event_name}', model_grp must be specified. "
                f"Available types: {available_types}. "
                f"Example: get_model_dir('{event_name}', model_grp='Amsterdam')"
            )

        # Normalize model_grp (case-insensitive)
        model_grp_capitalized = model_grp.capitalize()

        if model_grp_capitalized not in NICER_EVENT_PRESETS[nicer_canonical]:
            available_types = list(NICER_EVENT_PRESETS[nicer_canonical].keys())
            raise ValueError(
                f"Model type '{model_grp}' not found for pulsar '{event_name}'. "
                f"Available types: {available_types}"
            )

        preset_path = NICER_EVENT_PRESETS[nicer_canonical][model_grp_capitalized]
        inference_dir = Path(__file__).parent.parent
        model_dir_abs = (inference_dir / preset_path).resolve()

        logger.warning(
            f"No model_dir provided for NICER pulsar '{event_name}'. "
            f"Using default {model_grp_capitalized} preset path: {model_dir_abs}"
        )
        return str(model_dir_abs)

    # Event not found in any presets
    raise ValueError(
        f"No model_dir provided for event '{event_name}' and event is not in presets. "
        f"Available GW presets: {list(GW_EVENT_PRESETS.keys())}. "
        f"Available NICER presets: {list(NICER_EVENT_PRESETS.keys())}. "
        f"Please provide model_dir explicitly in the configuration."
    )


def create_likelihood(
    config: LikelihoodConfig,
):
    """
    Create likelihood from configuration

    Parameters
    ----------
    config : LikelihoodConfig
        Likelihood configuration

    Returns
    -------
    LikelihoodBase or None
        Configured likelihood instance, or None if disabled
    """
    if not config.enabled:
        return None

    params = config.parameters

    if config.type == "gw":
        # GW likelihoods are handled specially in create_combined_likelihood
        # This function should not be called directly for GW type
        raise RuntimeError(
            "GW likelihoods should be created via create_combined_likelihood, "
            "not create_likelihood directly"
        )

    elif config.type == "nicer":
        # NICER likelihoods are handled specially in create_combined_likelihood
        # This function should not be called directly for NICER type
        raise RuntimeError(
            "NICER likelihoods should be created via create_combined_likelihood, "
            "not create_likelihood directly"
        )

    elif config.type == "radio":
        # Radio timing likelihoods are handled specially in create_combined_likelihood
        # This function should not be called directly for radio type
        raise RuntimeError(
            "Radio timing likelihoods should be created via create_combined_likelihood, "
            "not create_likelihood directly"
        )

    elif config.type == "chieft":
        return ChiEFTLikelihood(
            low_filename=params.get("low_filename", None),
            high_filename=params.get("high_filename", None),
            nb_n=params.get("nb_n", 100),
        )

    elif config.type == "rex":
        experiment_name = params.get("experiment_name", "PREX")

        # FIXME: Implement load_rex_posterior(experiment_name) -> gaussian_kde
        # This should load PREX/CREX posterior KDE from data files
        # For now, raise NotImplementedError
        raise NotImplementedError(
            f"REX likelihood data loading not implemented. "
            f"Need to implement load_rex_posterior('{experiment_name}') -> gaussian_kde"
        )

    elif config.type == "constraints_eos":
        return ConstraintEOSLikelihood(
            penalty_causality=params.get("penalty_causality", -1e10),
            penalty_stability=params.get("penalty_stability", -1e5),
            penalty_pressure=params.get("penalty_pressure", -1e5),
        )

    elif config.type == "constraints_tov":
        return ConstraintTOVLikelihood(
            penalty_tov=params.get("penalty_tov", -1e10),
        )

    elif config.type == "constraints_gamma":
        return ConstraintGammaLikelihood(
            penalty_gamma=params.get("penalty_gamma", -1e10),
        )

    elif config.type == "zero":
        return ZeroLikelihood()

    else:
        raise ValueError(f"Unknown likelihood type: {config.type}")


def create_combined_likelihood(
    likelihood_configs: list[LikelihoodConfig],
):
    """
    Create combined likelihood from list of configs

    Parameters
    ----------
    likelihood_configs : list[LikelihoodConfig]
        List of likelihood configurations

    Returns
    -------
    LikelihoodBase
        Combined likelihood or single likelihood

    Raises
    ------
    ValueError
        If no likelihoods are enabled
    """
    likelihoods = []

    for config in likelihood_configs:
        if not config.enabled:
            continue

        # Special handling for GW likelihoods (presampled is now default): create one likelihood per event
        if config.type == "gw":
            params = config.parameters
            events = params["events"]  # Required, validated by schema
            penalty_value = params.get("penalty_value", -99999.0)
            N_masses_evaluation = params.get("N_masses_evaluation", 2000)
            N_masses_batch_size = params.get("N_masses_batch_size", 1000)
            seed = params.get("seed", 42)

            # Create one GWLikelihood (presampled) per event
            for event in events:
                # Get model directory (use preset if not provided)
                model_dir = get_model_dir(
                    event_name=event["name"], model_dir=event.get("model_dir")
                )

                gw_likelihood = GWLikelihood(
                    event_name=event["name"],
                    model_dir=model_dir,
                    penalty_value=penalty_value,
                    N_masses_evaluation=N_masses_evaluation,
                    N_masses_batch_size=N_masses_batch_size,
                    seed=seed,
                )
                likelihoods.append(gw_likelihood)

        # Special handling for GW likelihoods with resampling: create one likelihood per event
        elif config.type == "gw_resampled":
            params = config.parameters
            events = params["events"]  # Required, validated by schema
            penalty_value = params.get("penalty_value", -99999.0)
            N_masses_evaluation = params.get("N_masses_evaluation", 20)
            N_masses_batch_size = params.get("N_masses_batch_size", 10)

            # Create one GWLikelihoodResampled per event
            for event in events:
                # Get model directory (use preset if not provided)
                model_dir = get_gw_model_dir(
                    event_name=event["name"], model_dir=event.get("model_dir")
                )

                gw_likelihood = GWLikelihoodResampled(
                    event_name=event["name"],
                    model_dir=model_dir,
                    penalty_value=penalty_value,
                    N_masses_evaluation=N_masses_evaluation,
                    N_masses_batch_size=N_masses_batch_size,
                )
                likelihoods.append(gw_likelihood)

        # Special handling for NICER likelihoods: create one likelihood per pulsar
        elif config.type == "nicer":
            params = config.parameters
            pulsars = params["pulsars"]  # Required, validated by schema
            N_masses_evaluation = params.get("N_masses_evaluation", 100)
            N_masses_batch_size = params.get("N_masses_batch_size", 20)
            penalty_value = params.get("penalty_value", -99999.0)

            # Create one NICERLikelihood per pulsar
            for pulsar in pulsars:
                model_dir_ams = get_model_dir(
                    event_name=pulsar["name"],
                    model_dir=pulsar.get("amsterdam_model_dir"),
                    model_grp="Amsterdam",
                )
                model_dir_mry = get_model_dir(
                    event_name=pulsar["name"],
                    model_dir=pulsar.get("maryland_model_dir"),
                    model_grp="Maryland",
                )
                nicer_likelihood = NICERLikelihood(
                    psr_name=pulsar["name"],
                    amsterdam_model_dir=model_dir_ams,
                    maryland_model_dir=model_dir_mry,
                    N_masses_evaluation=N_masses_evaluation,
                    N_masses_batch_size=N_masses_batch_size,
                    penalty_value=penalty_value
                )
                likelihoods.append(nicer_likelihood)

        # Special handling for radio timing likelihoods: create one likelihood per pulsar
        elif config.type == "radio":
            params = config.parameters
            pulsars = params["pulsars"]  # Required, validated by schema
            penalty_value = params.get("penalty_value", -1e5)

            # Create one RadioTimingLikelihood per pulsar
            for pulsar in pulsars:
                radio_likelihood = RadioTimingLikelihood(
                    psr_name=pulsar["name"],
                    mean=pulsar["mass_mean"],
                    std=pulsar["mass_std"],
                    penalty_value=penalty_value,
                )
                likelihoods.append(radio_likelihood)

        else:
            # For other likelihoods, use standard creation
            likelihood = create_likelihood(config)
            if likelihood is not None:
                likelihoods.append(likelihood)

    if len(likelihoods) == 0:
        raise ValueError("No likelihoods enabled in configuration")
    elif len(likelihoods) == 1:
        return likelihoods[0]
    else:
        return CombinedLikelihood(likelihoods)
