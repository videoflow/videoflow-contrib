# Deploying and verifying the solutions

How to take the solutions in `solutions/` and prove they actually run as distributed flows —
locally, and on a Kubernetes cluster. This is the command-level companion to
[`../agents/solution-verifier.md`](../agents/solution-verifier.md), which holds the judgement —
what to try first, how to tell the failure layers apart, and when to stop.

> Keep this file in sync with the deploy CLI in
> [`../../../videoflow/videoflow/deploy/cli.py`](../../../videoflow/videoflow/deploy/cli.py) and
> with each solution's `config.template.yaml` and `README.md`. Nothing here describes one
> machine: every cluster fact is a probe you run, and every per-cluster value lives in the
> cluster profile file.

## Preconditions — probes, not assumptions

Each is seconds. An image build is minutes. Run them before building anything.

```bash
videoflow --help                                     # the CLI, installed from the sibling checkout
python -c "import videoflow, os; print(os.path.dirname(os.path.dirname(videoflow.__file__)))"
                                                     # ... and from SOURCE: that is the checkout the
                                                     #     base image is built from
docker info --format '{{json .Runtimes}}' | grep -q nvidia && echo "nvidia runtime" || echo "no nvidia runtime (GPU workers run device-less locally)"

kubectl config current-context                       # the cluster you mean — deploy never switches it
kubectl get nodes -o wide                            # one node: side-loading works; several: you need a registry
python -c "from videoflow.deploy.cluster import detect_cluster; print(detect_cluster())"
kubectl get runtimeclass                             # an `nvidia` class is applied to GPU pods automatically
kubectl get storageclass                             # an RWX class, for a multi-node work claim
kubectl get nodes -l videoflow.io/gpu-pool=true      # only for GPU flows
```

For a multi-node cluster, two more:

```bash
curl -sf "http://<registry>/v2/_catalog"             # the registry the nodes pull from answers
kubectl run vf-egress --rm -i --restart=Never --image=videoflow-base:py3.12 --command -- \
    python -c "import urllib.request; print(urllib.request.urlopen('https://github.com', timeout=8).status)"
                                                     # can pods reach the internet? if not, the prepare
                                                     # hook's caches (mount_home) are all the pods get
```

## The cluster profile

A single-node laptop cluster (kind, minikube, k3s, Docker Desktop) needs no profile. Anything
else gets one entry in `~/.config/videoflow/clusters.yaml`, keyed by the kubectl context it
describes; `deploy` and `teardown` take their defaults from it (an explicit flag still wins), and
say so. Example for a four-node k3s cluster with a plain-HTTP registry on the control plane, an
NFS-backed RWX StorageClass and a shared PriorityClass:

```yaml
docker:                                   # machine-level, every docker build / run (optional)
  build_args: ''
clusters:
  lab:
    context: default                      # kubectl config current-context
    namespace: videoflow
    registry: 10.0.0.1:5000               # push with crane: the docker daemon does not trust plain HTTP
    push_tool: crane
    mount_pvc: ['vf-share:/opt/data/cluster-share/pvc-<id>']   # the claim, at its backing directory
    mount_home: /opt/data/cluster-share/pvc-<id>/home           # the caches, inside the claim
    priority_class: cluster-batch
    # gpu_nodes: [gpu-01]                 # only the GPU nodes that are yours
    broker_profile: durable               # the nodes' root disks are ~8 GB: a video-sized Redis
    broker_storage_class: nfs-shared      # append-only file on an emptyDir gets the pod evicted,
    broker_replicas: 1                    # so broker and store live on claims of the RWX class
```

The claim: create an RWX PersistentVolumeClaim in the namespace (see `k8s/test-pvc.yaml` in the
core repo for the shape) and find the directory it is served at on your machine — for the NFS
CSI driver, `<share>/<subdir>` from the bound PersistentVolume:

```bash
kubectl create namespace videoflow
kubectl apply -n videoflow -f - <<'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata: {name: vf-share, labels: {app.kubernetes.io/managed-by: videoflow}}
spec: {accessModes: [ReadWriteMany], storageClassName: nfs-shared, resources: {requests: {storage: 50Gi}}}
EOF
pv=$(kubectl get pvc -n videoflow vf-share -o jsonpath='{.spec.volumeName}')
kubectl get pv "$pv" -o jsonpath='{.spec.csi.volumeAttributes.share}/{.spec.csi.volumeAttributes.subdir}{"\n"}'
```

That directory must be readable and writable on your machine (mount the export, or run on the
host that serves it). Everything the solution reads or writes goes under it: answer the
`work_dir` question with `<dir>/<solution>/out`, point your own input videos there, and let
`mount_home` put the model caches there — the prepare container on your machine fills them, the
pods mount them from the claim.

## The solutions

| Solution | Dockerfile | GPU | Prep fetches | Success artifact |
|---|---|---|---|---|
| `face_obfuscation` | `Dockerfile` (CPU by default) | 0 (1 for the detector with `device: gpu`) | sample clip, face SSD weights | `<work_dir>/blurred_video.avi` |
| `human_tracking` | `Dockerfile` (CPU by default) | 0 (1 each for pose and encoder with `device: gpu`) | sample clip, encoder and pose weights | `<work_dir>/annotated_video.avi` |

Both run with every default answered by Enter: the bundled sample clip, CPU, batch. The core
repo's `toy_*` solutions are the cheapest first target on a new cluster — they build in seconds
and need no weights — and `videoflow run-local` on one of them proves the framework path with no
cluster at all.

## Local run

```bash
cd solutions/human_tracking
videoflow run-local human_tracking.py --non-interactive      # with a config.yaml in place
```

The graph does not import on the host (no torch here), so run-local builds the solution image
(and `videoflow-base` before it), runs `prepare.py` and the compile inside it, starts a dev
NATS + Redis in docker unless something already listens on 4222/6379, and runs every worker as
a container of the image on the host network (`docker ps` shows them as `vf-<flow>-<run>-<node>-<replica>`).
Success: exit 0, `Flow <id> completed.`, and a fresh, non-empty `out/annotated_video.avi`
(`stat -c '%y %s' out/annotated_video.avi`). Files written by the containers are root-owned.

## Cluster deploy

```bash
cd solutions/human_tracking
videoflow deploy human_tracking.py --non-interactive --flow-id human-tracking --run-id ht-1
```

Everything else comes from the profile (or the defaults, on a laptop cluster). The flags:

| Flag | Why |
|---|---|
| `--non-interactive` | No TTY. Gives a useful `SystemExit` listing required inputs when `config.yaml` is missing. |
| `--flow-id` + `--run-id` | Deterministic, and what `teardown` and every `-l videoflow.io/run-id=` selector need. DNS-1123-safe hyphens, never `human_tracking`. Increment the run-id every attempt; never reuse one. |
| `--keep-infra` | Amortises NATS+Redis across runs instead of paying recreation each time. |
| *(no `--keep` by default)* | Deploy dumps failed nodes' logs before teardown, and auto-teardown frees the GPUs. Add `--keep` only for a deliberate diagnostic re-run, then tear down immediately. |

What deploy does with it: builds the image the template's `x-gpu` selects, tags it by content,
pushes it to the profile's registry (or side-loads it on a single-node cluster), runs
`prepare.py` and the compile inside it, puts the cluster's `nvidia` RuntimeClass on GPU pods,
provisions the dev broker in the namespace (reused when present), applies, waits for the BATCH
flow, prints `Flow <id> completed.`, tears the run down.

Wrap it in `timeout` (900s for the CPU solutions). A pod stuck in `ImagePullBackOff` is not
"Unschedulable", so the 60s watchdog never fires and the wait can hang; treat a timeout as a
triage signal, not a crash. Long builds and runs exceed the foreground command timeout — run
them in the background with output tee'd to a log, then poll.

**Success is a conjunction:**

```bash
grep -q 'Flow human-tracking completed\.' deploy.log                       # 1. the completion line
kubectl get jobs -n videoflow -l videoflow.io/run-id=ht-1                   # 2. nothing left behind
find <work_dir> -name annotated_video.avi -newermt '-30 minutes' -size +0   # 3. a FRESH, non-empty artifact
```

Point 3 needs the mtime guard specifically: prep and compile run as root in-image, so a previous
run's artifact is root-owned and cannot be deleted without sudo. Existence alone is not evidence.

## Offline validation — before a cluster deploy you cannot afford to repeat

```bash
videoflow deploy human_tracking.py --dry-run --non-interactive > render.yaml
```

With the image built, deploy compiles **inside the solution image** (the graph dir mounted at
the same absolute path), so a clean dry run has already proven: the config parses, prep
artifacts resolve, contrib imports, every node's `get_params()` contract holds, the graph
compiles, and the manifests render — with no cluster. A render never touches the cluster, so it
sets no RuntimeClass by itself; pass `--gpu-runtime-class nvidia` to see the GPU pods as the
live deploy renders them. Then check that every path-valued entry of each node's
`VF_NODE_PARAMS_JSON` falls under one of that workload's `volumeMounts` (or a claim `subPath`).

**Do not use `videoflow explain` for this** — it only compiles on the host, where
`videoflow_contrib` isn't installed, so it always fails here.

## Observing a run

There is **no `videoflow status` and no `videoflow logs`.** Observation is raw `kubectl` plus the
NATS CLI; the only built-in capture is the automatic dump when a BATCH node fails.

```bash
kubectl get pods -n videoflow -l videoflow.io/run-id=<run> -o wide
kubectl logs -n videoflow -l videoflow.io/run-id=<run> --tail=200 --all-containers --prefix
kubectl describe pod <pod> -n videoflow            # names Pending / ImagePull reasons directly
videoflow debug decode --dlq --flow-id <flow> --run-id <run> --nats <url> --limit 20

videoflow dlq ls --flow-id <flow>                  # triage: dead letters grouped by code
videoflow dlq show --flow-id <flow> --id 3         # one entry, payload included
```

Every resource carries `videoflow.io/flow-id`, `videoflow.io/run-id` and `videoflow.io/node`, so
those selectors scope cleanly to one run.

Health, on container port 8080: `/readyz` returns 200 only after the node's `open()` returns — a
pod stuck not-ready means `open()` is hanging or failing. `/healthz` goes 503 after 60s without a
beat. `/metrics` carries `videoflow_messages_{published,received,processed,failed}_total{node=…}`,
which is how you tell a flow that is working from one that is merely up, plus
`videoflow_errors_total{node,code,disposition}`, which is how you tell *what* is failing.

**Triage failures by code, not by log volume.** `videoflow dlq ls` groups by the stable code:

| What you see | What it means | Where the fix is |
|---|---|---|
| exit 2 / `VF_CONFIG`, `VF_CAPABILITY` | a node param or the config is wrong | this repo — the graph or `config.yaml` |
| exit 3 / `VF_RESOURCE_UNAVAILABLE`, `VF_CLUSTER` | the world is wrong | the cluster, the footage, the registry |
| many `VF_POISON_SCHEMA` from one node | its upstream is emitting the wrong shape | the producing component's output contract |
| any `VF_DEVICE` / `VF_RESOURCE_EXHAUSTED` | a sick worker, not bad data | GPU sizing — and check the component registered a classifier for its framework's OOM type |

A stream of dead letters whose code has nothing to do with their payloads is the signature of a
missing classifier registration: the failure is being read as transient and retried into the DLQ.

## Teardown

```bash
videoflow teardown --flow-id human-tracking --run-id ht-1 --nats <url>        # namespace from the profile

# once, at the very end (removes the dev NATS/Redis deploy provisioned):
videoflow teardown --flow-id human-tracking --run-id ht-N --nats <url> --infra
```

`--flow-id` and `--run-id` are required; `--nats` too unless the profile names the broker
(**capture the URL from the teardown hint deploy prints** rather than guessing it); `--infra`
additionally requires a namespace. Carry `--gpu-mode` when deploy printed it — teardown is the
only place a GPU strategy's `cleanup()` runs on the REALTIME path.

Always tear a run down before advancing to the next run-id. Teardown is run-scoped, so it won't
disturb a concurrent run — but **omitting `--run-id` from a manual label delete removes every run
of the flow.** A pre-existing broker is reused and never deleted, by design in both `infra.py`
and `localinfra.py`.
