#!/bin/sh

if [ "$#" -ne 1 ] || [ -z "$1" ]; then
	printf 'Usage: %s DIRECTORY\n' "$0" >&2
	exit 2
fi

root=$1
while [ "$root" != / ] && [ "${root%/}" != "$root" ]; do
	root=${root%/}
done
case $root in
/*) ;;
*) root=./$root ;;
esac

if [ -L "$root" ] || [ ! -d "$root" ]; then
	printf '%s: expected a non-symlink directory: %s\n' "$0" "$1" >&2
	exit 2
fi

exec find -P "$root" -type f -exec cat -- {} + >/dev/null
