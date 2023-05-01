hwpwn 0.1.5 (2023-05-01)
========================

Bugfixes
--------

- Bug fix in GitHub action for package release.


hwpwn 0.1.4 (2023-05-01)
========================

Features
--------

- Added `config_set` function to common module.
- Added `data_out` configuration parameter for controlling if data is sent to output or not.
- The release description in GitHub should now include the last CHANGELOG contents.


Bugfixes
--------

- Bug fix in typer callback usage to avoid duplicate calls of callback.


hwpwn 0.1.3 (2023-04-30)
========================

Features
--------

- Added dpi argument to time-series plot command.


Bugfixes
--------

- The `load` function should return the data loaded.
- The `process_raw_table_signals` was returning the wrong variables.
