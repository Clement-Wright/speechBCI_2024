# speechBCI Runtime Helper

This repository vendors a lightweight wrapper around the
[`speechBCI`](https://github.com/fwillett/speechBCI) language-model decoder.
To keep the repository size manageable, the compiled runtime assets are not
committed directly. Instead, run the helper below to download the upstream
artifacts when needed:

```bash
python -m third_party.speechBCI.runtime --quiet --force
```

The command fetches the latest `LanguageModelDecoder/runtime/server/x86`
contents from the upstream repository and places them under
`third_party/speechBCI/runtime/server/x86`. Once the download finishes,
the `LanguageModelDecoderAdapter` can locate the WFST decoder libraries
without any further configuration.

If you prefer to pin a specific revision, pass the `--repo-zip-url` flag
with a direct link to a GitHub archive (for example, a release zip or a
commit snapshot).
