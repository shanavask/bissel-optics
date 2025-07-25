

Configure project

```
gcloud iam service-accounts create vertex-pipeline-sa \
    --description="run vertex pipelines" \
    --display-name="vertex-pipeline-sa" \
    --project=jfgp-creative-optics
```

```
gcloud projects add-iam-policy-binding jfgp-creative-optics \
    --member="serviceAccount:vertex-pipeline-sa@jfgp-creative-optics.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

```
gcloud iam service-accounts add-iam-policy-binding \
    vertex-pipeline-sa@jfgp-creative-optics.iam.gserviceaccount.com \
    --member="user:shanavas.kavu@jellyfish.com" \
    --role="roles/iam.serviceAccountUser"
```

or

```
gcloud projects add-iam-policy-binding jfgp-creative-optics \
    --member="serviceAccount:993945077455-compute@developer.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

```
gcloud storage buckets create gs://jfgp-creative-optics-pipe --location=us-central1 --project=jfgp-creative-optics
```

```
gcloud storage buckets add-iam-policy-binding gs://jfgp-creative-optics-pipe \
--member=serviceAccount:vertex-pipeline-sa@jfgp-creative-optics.iam.gserviceaccount.com \
--role=roles/storage.objectUser
```

```
gcloud storage buckets add-iam-policy-binding gs://bkt-bissell-prod-us-creative \
--member=serviceAccount:vertex-pipeline-sa@jfgp-creative-optics.iam.gserviceaccount.com \
--role=roles/storage.objectAdmin
```

or

```
gcloud storage buckets add-iam-policy-binding gs://jfgp-creative-optics-pipe \
--member=serviceAccount:993945077455-compute@developer.gserviceaccount.com \
--role=roles/storage.objectUser
```

read access to bq table

```
bq add-iam-policy-binding \
    --member="serviceAccount:vertex-pipeline-sa@jfgp-creative-optics.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataViewer" \
    rp-tech-bissell-prod:funnel_export_bissell_us.Creative_Report_Meta
```

```
bq add-iam-policy-binding \
    --member="serviceAccount:vertex-pipeline-sa@jfgp-creative-optics.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataEditor" \
    rp-tech-bissell-prod:optics
```

```
gcloud projects add-iam-policy-binding jfgp-creative-optics \
    --member="serviceAccount:vertex-pipeline-sa@jfgp-creative-optics.iam.gserviceaccount.com" \
    --role="roles/bigquery.jobUser"
```

```
gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET_NAME \
    --member="serviceAccount:vertex-pipeline-sa@jfgp-creative-optics.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```