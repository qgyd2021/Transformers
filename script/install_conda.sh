#!/usr/bin/env bash

# https://www.5axxw.com/questions/simple/umiecs


# params:
system_version="centos";


# parse options
while true; do
  [ -z "${1:-}" ] && break;  # break if there are no arguments
  case "$1" in
    --*) name=$(echo "$1" | sed s/^--// | sed s/-/_/g);
      eval '[ -z "${'"$name"'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;
      old_value="(eval echo \\$$name)";
      if [ "${old_value}" == "true" ] || [ "${old_value}" == "false" ]; then
        was_bool=true;
      else
        was_bool=false;
      fi

      # Set the variable to the right value-- the escaped quotes make it work if
      # the option had spaces, like --cmd "queue.pl -sync y"
      eval "${name}=\"$2\"";

      # Check that Boolean-valued arguments are really Boolean.
      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
        exit 1;
      fi
      shift 2;
      ;;

    *) break;
  esac
done


echo "system_version: ${system_version}";


if [ ${system_version} = "centos" ]; then
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

  bash Miniconda3-latest-Linux-x86_64.sh

  /usr/local/miniconda3/bin/conda --version

  cat ~/.bashrc
  echo "PATH=$PATH:/usr/local/miniconda3/bin" >> /root/.bashrc
  source ~/.bashrc

  conda --version

fi



