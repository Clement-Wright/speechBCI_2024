import logging
from typing import Iterable, Optional, Tuple

from torch.nn import Module


def load_state_dict_compat(
    model: Module,
    state_dict,
    *,
    strict: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Iterable[str], Iterable[str]]:
    """Load a checkpoint while tolerating missing keys.

    Falls back to ``strict=False`` so checkpoints saved before architectural
    changes (e.g., missing convolutional branches or attention blocks) remain
    usable.  Missing and unexpected keys are logged to help with debugging.
    """

    if any(key.startswith("fc_decoder_out") for key in state_dict.keys()):
        state_dict = state_dict.__class__(state_dict)  # preserve OrderedDict
        for key in list(state_dict.keys()):
            if key.startswith("fc_decoder_out"):
                new_key = key.replace(
                    "fc_decoder_out", "decoder_adapter.projection", 1
                )
                state_dict[new_key] = state_dict.pop(key)

    try:
        result = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        result = model.load_state_dict(state_dict, strict=False)
        log = logger if logger is not None else logging.getLogger(__name__)
        if result.missing_keys:
            log.warning(
                "Missing parameters when loading checkpoint: %s",
                result.missing_keys,
            )
        if result.unexpected_keys:
            log.warning(
                "Unexpected parameters when loading checkpoint: %s",
                result.unexpected_keys,
            )
    return getattr(result, "missing_keys", ()), getattr(
        result, "unexpected_keys", ()
    )


__all__ = ["load_state_dict_compat"]
