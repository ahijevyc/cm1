import os

import f90nml


class RunController:
    def __init__(self, file_path):
        """
        Initialize the controller with a CM1 run configuration file.
        """
        self.file_path = file_path
        self.config = f90nml.read(file_path)
        self.walltime = "02:00:00"
        self.select = 1
        self.account = "NMMM0021"
        self.needLW = False
        self.needSW = False
        self.executable = "cm1.exe"
        self.input_sounding = None
        self.run_dir = None
        self.name = None

class FortranNamelistController:
    def __init__(self, file_path):
        """
        Initialize the controller with a Fortran namelist file.
        """
        self.file_path = file_path
        self.namelist = f90nml.read(file_path)

    def get_value(self, group, variable):
        """
        Get the value of a variable in a specific namelist group.
        """
        try:
            return self.namelist[group][variable]
        except KeyError:
            raise KeyError(
                f"Group '{group}' or variable '{
                           variable}' not found."
            )

    def set_value(self, group, variable, value):
        """
        Set the value of a variable in a specific namelist group.
        """
        if group in self.namelist:
            self.namelist[group][variable] = value
        else:
            raise KeyError(f"Group '{group}' not found.")

    def add_group(self, group):
        """
        Add a new group to the namelist.
        """
        if group not in self.namelist:
            self.namelist[group] = {}
        else:
            raise ValueError(f"Group '{group}' already exists.")

    def add_variable(self, group, variable, value):
        """
        Add a variable to a specific namelist group.
        """
        if group not in self.namelist:
            raise KeyError(f"Group '{group}' not found.")
        self.namelist[group][variable] = value

    def delete_variable(self, group, variable):
        """
        Delete a variable from a specific namelist group.
        """
        try:
            del self.namelist[group][variable]
        except KeyError:
            raise KeyError(
                f"Group '{group}' or variable '{
                           variable}' not found."
            )

    def save(self, output_file, force=False):
        """
        Save the updated namelist to a file. By default, this will not overwrite an existing file.
        """
        f90nml.write(self.namelist, output_file, force=force)
        print(f"Namelist saved to {output_file}")