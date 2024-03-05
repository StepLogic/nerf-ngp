#ssh grtx-1
module load singularity
export TMPDIR=$XDG_RUNTIME_DIR
singularity build --fakeroot $HOME/object_nerf.simg object_nerf.def