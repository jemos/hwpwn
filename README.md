# hwpwn

A user-friendly tool designed to facilitate hardware security attacks. The aim is to provide a simple interface with 
commands that cover multiple steps typically involved in hardware security attacks. Plus, it makes repeating attacks 
or changing details very easy, so you can quickly see what happens when you tweak something.

# Quickstart

First install the package with pip:

    $ pip install hwpwn --user

The hwpwn has multiple commands which can be called in chain, using shell pipe. Alternatively, you can write a flow
YAML file which has the commands described in textual format and run the flow with hwpwn client.

Create a dummy data file:

    $ cat <<EOF > test.csv
    t,a,b,c
    1,3,3,4
    2,3,4,5
    3,9,2,1
    EOF

To use in piped commands one can, for example:

    $ hwpwn data load test.csv | hwpwn data subtract a b ab --append | cat
    {"x_axis": [1.0, 2.0, 3.0], "signals": [{"name": "a", "vector": [3.0, 3.0, 9.0]},
    {"name": "b", "vector": [3.0, 4.0, 2.0]}, {"name": "c", "vector": [4.0, 5.0, 1.0]},
    {"name": "ab", "vector": [-1.0, -2.0, 8.0]}], "triggers": [], "ts": 1}

If we wanted to execute these commands in a flow, we needed to create a flow YAML, let's say, `my_flow.yaml`:

    ---
    options:
      scale: 1e-6
      ts: 4e-9
      description: |
        My long description. It can take a single line or multiple lines.
        Line number two.
    operations:
      - data.load:
          filepath: test.csv
      - data.subtract:
          pos: a
          neg: b
          dest: ab
          append: true

Now run hwpwn with:

    $ hwpwn --verbose flow run test.yaml | cat
    INFO: trying to load file test.csv ...
    INFO: loaded 3 datapoints from test.csv.
    INFO: found signal a
    INFO: found signal b
    INFO: found signal c
    INFO: using sampling period of 4.0e-09 [s].
    INFO: calculating signal subtract ab = a - b (append=True)
    {"x_axis": [1.0, 2.0, 3.0], "signals": [{"name": "a", "vector": [3.0, 3.0, 9.0]}, 
    {"name": "b", "vector": [3.0, 4.0, 2.0]}, {"name": "c", "vector": [4.0, 5.0, 1.0]}, 
    {"name": "ab", "vector": [0.0, -1.0, 7.0]}], "triggers": [], "ts": 4e-09}

I've used the `--verbose` flag which will make it show what is happening. If you don't pipe the output of hwpwn,
it will not show the final data to avoid flooding the console. Thus, I've used a piped `cat` command.

# Documentation

The documentation is built with Sphinx and deployed in Github Pages. The URL is https://jemos.github.io/hwpwn. 

# Final Words

This project is currently in its early stages of development, and as such, it may not be as comprehensive or
polished as desired. However, it is a work in progress, and future enhancements are expected to improve
its functionality and usability in hardware security analysis. Users are encouraged to provide feedback
and contribute to the project's growth and refinement.

# Next Features...

Things I'd like to add in the future...

  * `scope` commands for interacting with USB osciloscopes
  * `spa` commands for simple power analysis
  * `dpa` commands for differential power analysis
  * `cpa` commands for correlation power analysis

