#!/usr/bin/env bash

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
  mkdir -p /data/dep
  cd /data/dep || exit 1;

  if [ ! -e openssl-1.1.1n.tar.gz ]; then
    wget https://www.openssl.org/source/openssl-1.1.1n.tar.gz --no-check-certificate
  fi

  cd /data/dep || exit 1;
  if [ ! -d openssl-1.1.1n ]; then
    tar -zxvf openssl-1.1.1n.tar.gz

    cd /data/dep/openssl-1.1.1n || exit 1;
  fi

  ./Configure --prefix=/usr/local/openssl

  make -j && make install

fi
