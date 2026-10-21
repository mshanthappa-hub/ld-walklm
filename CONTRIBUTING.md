<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Contribution policy

This project is not currently accepting external code or documentation
contributions. No project CLA is required while that policy remains in effect.
This policy governs acceptance into this repository; it does not change the
rights granted to recipients under the applicable licenses.

Before enabling external contributions, maintainers must obtain the approved
NVIDIA Contributor License Agreement, publish the exact agreement and signing
instructions, verify the required agreement for each contributor, and update
this policy. Do not put signed agreements or personal information in public
issues or pull requests.

## Maintainer changes

- Perform a self-review, then an independent peer IP review before distribution.
  Follow the applicable NVIDIA review process for ongoing changes.
- Review all changed files, including documentation, patches, inputs, and
  generated artifacts, for confidential material and third-party rights.
- Refer potentially patentable material to IP Legal before public disclosure.
  A license header is not patent or release approval.
- Preserve third-party copyright and license notices. Do not replace upstream
  ownership notices with NVIDIA notices.
- For NVIDIA-authored Python and shell source, use this approved SPDX form,
  with the correct creation year:

  ```text
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: MIT
  ```

  Keep a script's shebang on the first line. Use appropriate comment syntax
  for documentation and other source formats. In mixed-origin files, retain the
  full third-party notice independently of the NVIDIA notice for modifications.
- Update [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and include required
  license texts when introducing or updating third-party material.
- Confirm exact packages and all bundled/transitive notices for any
  distributed environment, wheel, container, or executable.
- Keep recorded experiment results and protocol evidence unchanged.
- Complete applicable review and release approvals before public release.

See [licensing scope](docs/LICENSING.md) for the distinction between project
source, third-party code, models, data, and generated artifacts.
