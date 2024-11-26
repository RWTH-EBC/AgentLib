# Changelog

## 0.8.6

- Fix a bug, where pandas series could not be sent over network communicators or when
  parsing json in local communication.

## 0.8.5

- AgentLogger now always needs a filename. However, the MAS utility variable_logging
  still autogenerates the file. The AgentLogger now creates all directories needed for
  the file. This makes it easier to save logs of larger agent systems in subdirectories.
  AgentLogger now also has the option 'overwrite_log' which checks whether existing log
  files should be overwritten when starting the agent system from scratch.

## 0.8.4

- Fixed a bug, where modules could deregister callbacks of other modules

## 0.8.3

- Add CSV Data source module. Allows to gradually send data from a csv file. Can be used
  to provide the simulator with variable data.
- Add optional dependecy interactive, which introduces dash. Can be used to create a
  dashboard of simulator results, updating in real time. Usage
    ```python
    from agentlib.utils import simulator_dashboard
    simulator_dashboard("results1.csv", "results2.csv")
  ```
  Make sure your simulation is already running and saving to a csv file.
- Fix causality print in writing csv data.

## 0.8.2

- Realtime Environment always has a clock to prevent agents that define callbacks only
  from terminating
- Environment time in real time is now based on system time, decoupling it from the
  simpy process
- Environment time in scaled real time is as before, and will be logged in seconds, not
  datetime
- Changed default t_sample for agentlogger to 60 seconds and added warning for
  performance if below that
- Change process error handling, so that exceptions are not caught by the process
  anymore, improving debugging. This might exclude the use of more advanced simpy
  features.
- Move pahomqtt dependency up to 2.0
- Fix a bug, where type validation was done incorrectly during initial configuration
- More specific optional dependency errors and correct docker image #21

## 0.8.1

- Simulator results now start at the real start, not after the first sample interval.
  Inputs and parameters are now written at the correct index in the results (they are
  one row too late before).

## 0.8.0

- Restructure DataBroker for real time execution. Each module now gets its own thread,
  preventing modules with slow callbacks from blocking other callbacks, e.g.
  communicators. Also ensure the entire agent fails, if there is an exception in a
  callback.

## 0.7.2

- fix an error, where custom injection would not work on modules including relative
  imports

## 0.7.1

- Fix an issue, where the shared attribute of varibles could not be set properly

## 0.7.0

- Migrated code to github

## 0.6.0

- Significant performance improvements
- Make many dependencies optional
- Restructure module to better use the configs for schema parsing
- Make compatible with pydantic v2

## 0.2.8

- Use unified subscription field for all communicators (#151)

## 0.2.7

- Remove deprecated code
- Remove string injection (#142)

## 0.2.6

- Make modular import possible (#121)

## v0.2.5

- Update the way as fast as possible callbacks work
- Add coordinated ADMM

## v0.2.4

- Add Fiware modules

## v0.2.3

- Remove causalities

## v0.2.2

- Enable real-time and as fast as possible callbacks

## v0.2.1

- Separate Module and ModuleConfig
- Change Cache to DataBroker

## v0.2.0

- Restructure modules by using separate Config and Function classes

## v0.1.0

- First implementation fixed by project milestone
