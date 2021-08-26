for i in {0..9};
do
echo "Worker $i";
sbatch -p low -w "cluster-worker-$i" --job-name "stop-$i" --wrap="srun gcloud compute instances stop cluster-worker-$i --quiet --zone europe-west1-b";
done