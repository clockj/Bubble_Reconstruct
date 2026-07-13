---
name: Rockfish HPC Workflow
description: Guidelines for working on Rockfish HPC cluster - USE ONLY when user mentions rockfish or remote server
---

# Rockfish HPC Workflow

**Only apply these rules when working on the Rockfish remote server.**

## Constraints

- Never use `sudo`, `systemctl`, or modify system configurations
- Never run multi-threaded or computationally intensive jobs locally
- Always use module system (`ml`) to load software

## Python Environment

```bash
ml anaconda
conda activate OED
```

## Installing Packages

1. First try: `ml <module_name>`
2. If not found:
   ```bash
   ml anaconda
   conda activate OED
   conda install <package>  # or pip install <package>
   ```

## Building C++ Code

```bash
conda deactivate
./command.rockfish
```

## Job Submission

Submit heavy computation to job scheduler. Check existing job scripts in project directory for templates.

## Commands

| Task | Command |
|------|---------|
| Load Anaconda | `ml anaconda` |
| Activate env | `conda activate OED` |
| Deactivate env | `conda deactivate` |
| Load module | `ml <name>` |
| List modules | `ml list` |
| Search modules | `ml spider <name>` |
| Build C++ | `./command.rockfish` |

