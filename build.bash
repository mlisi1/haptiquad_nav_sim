#!/usr/bin/bash
docker load --input anymal_sim/anymal_docker_image.tar
docker tag vsc-anydev-78cfdec13eca790f0f68fe9dc11138f8c58e3384251e24a5dbebd3625805170c-uid:latest anymal_sim_base:latest

