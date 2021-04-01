#cloud-config
cloud_final_modules:
- [scripts-user, always]
repo_upgrade: security

packages:
 - git

runcmd:
 - [cd, "/home/ec2-user"]
 - [ git, clone, "https://github.com/lincdog/webfish.git" ]
 - [ chown, -R, ec2-user, ./webfish ]
 - cd webfish
 - [ /opt/conda/bin/pip, install, -r, requirements.txt ]
