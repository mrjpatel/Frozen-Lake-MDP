# COSC1125 Artificial Intelligence - Assignment 3

# Setup

## Virtual Environment
Using `virtualenv` makes it easy to manage dependencies and isolate them from globally installed modules.

The `virtualenv` tool can be installed through `pip`. This will allow you to manage different virtual environments in the project.
```bash
$ sudo python3 -m pip install virtualenv
```

A new virtual environment can be created by giving `venv` the name of your virtual environment as a command line parameter. In this example, we are calling our environment `venv`.
```bash
$ python3 -m venv venv
```

Next we will have to activate our newly created environment.
```bash
$ source venv/bin/activate
```

Once activated, dependencies of the project will need to be installed. All dependencies are stored in the `requirements.txt` file, so this will need to be passed to `pip`.
```bash
$ pip install -r requirements.txt
```

Now you are ready to run python scripts in the project.

When you are done, deactivate the environment using the `deactivate` command.
```bash
$ deactivate
```

# Run Project
To tun the project, just run the script as shown below. This has code of value iteration and policy iteration.
```bash
$ python vi_and_pi.py
```
