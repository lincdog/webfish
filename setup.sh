#cloud-config
repo_upgrade: security

packages:
 - git

runcmd:
 - [ git, clone, "https://github.com/lincdog/webfish.git" ]
 - cd webfish
 - [ pip, install, -r, requirements.txt ]
