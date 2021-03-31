#cloud-config
repo_upgrade: security

packages:
 - git

runcmd:
 - conda activate
 - [cd, "/home/ec2-user"]
 - [ git, clone, "https://github.com/lincdog/webfish.git" ]
 - [ chown, -R, ec2-user, ./webfish ]
 - cd webfish
 - [ pip, install, -r, requirements.txt ]
