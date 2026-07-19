# Deploying and verifying the solutions

How to take the solutions in `solutions/` and prove they actually run as distributed flows on a
Kubernetes cluster. This is the command-level companion to
[`../agents/solution-verifier.md`](../agents/solution-verifier.md), which holds the judgement —
what to try first, how to tell the failure layers apart, and when to stop.

> Keep this file in sync with the deploy CLI in
> [`../../../videoflow/videoflow/deploy/cli.py`](../../../videoflow/videoflow/deploy/cli.py) and
> with each solution's `config.template.yaml` and `README.md`. The cluster facts below describe a
> specific machine and will drift — re-run the probes rather than trusting them.

## Preconditions — run these before building anything

Each is seconds. An image build is tens of minutes. **The expensive mistake in this job is
discovering an infrastructure blocker after a CUDA build**, so rule them out first.

```bash
kubectl config current-context                       # expect: k3s
kubectl get nodes -o wide
kubectl get runtimeclass nvidia                      # the opt-in GPU runtime
kubectl get nodes -l videoflow.io/gpu-pool=true -o name
kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}'   # expect 4

sudo -n true                                         # THE GATING PROBE — see below
df -h /                                              # images land twice: docker + containerd
docker system df

mkdir -p ~/.videoflow/models ~/.torch ~/.cache       # else root-owned containers create them
```

### The gating probe: can an image reach containerd?

`videoflow deploy` loads a locally-built image into k3s with
`docker save <img> | sudo k3s ctr images import -`
([`cluster.py`](../../../videoflow/videoflow/deploy/cluster.py)). That needs **passwordless
sudo**. On this machine it currently fails:

```
$ sudo -n true
sudo: a password is required
$ ls -l /run/k3s/containerd/containerd.sock
srw-rw---- 1 root root ...                    # and `ctr` against it also fails
```

An agent has no TTY to answer the prompt, so **image loading is blocked and no code change in
either repo fixes it.** The realistic unblocks are operator actions: a sudoers `NOPASSWD` entry
for `k3s ctr images import`, group ownership on the containerd socket, or a registry the node can
pull from (none is running, and `/etc/rancher/k3s/registries.yaml` needs root).

When this is the situation, **still run the whole offline half of the pipeline before reporting** —
everything through the `--dry-run` render below works without a cluster and proves far more than
"sudo failed".

### Cluster facts

| | |
|---|---|
| Context | `k3s` (current). A `kind-cluster` context also exists and is dead — ignore it. |
| Node | `jadiel-deep-learning`, single RTX 4090, already labeled `videoflow.io/gpu-pool=true` |
| GPU | `nvidia.com/gpu: 4` — **time-sliced from one physical card**, so no memory isolation |
| RuntimeClass | `nvidia` exists but is **opt-in**; runc stays the containerd default |

**`--gpu-runtime-class nvidia` is mandatory for any deploy with a GPU node.** Without it the pod
schedules happily and starts device-less, then fails deep inside model loading. It is the easiest
mistake to make here and it does not look like a flag problem.

Registered GPU modes are exactly `exclusive` (default) and `shared`; confirm with
`python -c "from videoflow.deploy.gpu import registered_gpu_modes; print(registered_gpu_modes())"`.
`shared` drops the resource limit, so with three detectors on one physical card it removes the only
thing stopping them piling onto one device. Prefer `exclusive` while demand ≤ 4.

## Bootstrap

The core CLI is already installed and the deploy extras are present (`nats-py`, `redis`, `yaml`,
`numpy`, `opencv`). `kubernetes` is absent and that is fine — the repo shells out to `kubectl` and
has no Python Kubernetes client. What is missing is any `videoflow-base` image: the daemon has
plenty of unrelated images, but none of ours.

```bash
cd /home/jadiel/workspace/videoflow
docker image inspect videoflow-base:py3.12-cuda >/dev/null 2>&1 || ./docker/build-images.sh
```

Then build each solution image **explicitly**, with the right Dockerfile and an immutable tag,
from the contrib repo root (solution Dockerfiles COPY sibling sub-packages):

```bash
cd /home/jadiel/workspace/videoflow-contrib
docker build -f solutions/offside/gpu.Dockerfile      -t videoflow-offside:r1          .
docker build -f solutions/human_tracking/Dockerfile   -t videoflow-human-tracking:r1   .
docker build -f solutions/face_obfuscation/Dockerfile -t videoflow-face-obfuscation:r1 .

# Smoke-test CUDA in the GPU image now, not inside a worker pod 40 minutes later.
docker run --rm --gpus all videoflow-offside:r1 \
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Never let `videoflow deploy` autobuild** — always `--no-build --image <ref>`:

- `autobuild` picks `gpu.Dockerfile` whenever the *docker daemon* exposes an nvidia runtime
  (`docker_gpus_available()` is `True` here), regardless of the flow's device placement. Left
  alone it builds a CUDA image for the two CPU-only solutions.
- `autobuild` tags `:latest`, and **core sets no `imagePullPolicy` anywhere**. Kubernetes defaults
  `:latest` to `Always`, so a locally-imported image is re-pulled from a registry that hasn't got
  it. It surfaces as `provision Job did not complete within 180s`, because the provision Job runs
  on the same image in phase 1 of the two-phase apply.

**Tags are immutable and increment on every rebuild** (`:r1`, `:r2`, …). A non-`:latest` tag gets
`IfNotPresent` — which is what makes a locally-loaded image usable at all, and exactly why reusing
a tag after a fix silently runs the **stale** image.

Do **not** install `videoflow_contrib` into the core venv. The components have mutually
incompatible dependencies (tensorflow vs torch) and the repo is deliberate about this. The
consequence is that `run-local` and `videoflow explain` are both unusable here — they compile on
the host. `--dry-run` replaces them and is strictly better (see below).

Record what you are testing; it belongs at the top of any report:

```bash
git -C /home/jadiel/workspace/videoflow         log -1 --format='core    %h %s'
git -C /home/jadiel/workspace/videoflow-contrib log -1 --format='contrib %h %s'
```

## Offline validation — do this before every cluster deploy

```bash
cd /home/jadiel/workspace/videoflow-contrib/solutions/offside
videoflow deploy offside.py \
    --dry-run --no-build --image videoflow-offside:r1 \
    --config config.yaml --non-interactive \
    --namespace videoflow --gpu-runtime-class nvidia \
    > /tmp/offside-render.yaml
```

This is the highest value-per-second step in the whole loop. With `--image` set, the host compile
fails on `ModuleNotFoundError` (an `ImportError`) and deploy falls back to compiling **inside the
solution image**, with the graph dir mounted at the same absolute path. So a clean dry run has
already proven: the config parses, prep artifacts resolve, contrib imports, every node's
`get_params()` contract holds, the graph compiles, and the manifests render — with no cluster.

Then assert on the rendered YAML:

```bash
grep -c 'runtimeClassName: nvidia' /tmp/offside-render.yaml   # offside: expect 3 detectors
grep -A2 'nvidia.com/gpu' /tmp/offside-render.yaml            # expect 3 x limit 1
```

and check that **every path-valued entry of each node's `VF_NODE_PARAMS_JSON` falls under one of
that workload's `volumeMounts`**. A path that doesn't is a `FileNotFoundError` in a pod later, and
this is the cheapest place to catch it.

Leave `--no-prepare` **off** on the first pass so the prep hook runs in-image and materialises
weights and sample clips; add it on every later iteration.

## Where the config must live

**`--config` never reaches `build_flow`.** `load_flow` calls `factory()` with no arguments, and
each solution's `build_flow(cfg=None)` then reads `os.path.dirname(__file__)/config.yaml`
directly. `--config` only feeds `x-mount` resolution and `prepare.py --config`.

So the config **must** be written to `<solution>/config.yaml`. Put it anywhere else and compile
reads a different file — and a missing one raises `FileNotFoundError`, which the compile path does
*not* catch (it only catches `ImportError`), so you get a raw traceback instead of the in-image
fallback. Pass `--config config.yaml` as well, for the mount resolution.

`solutions/*/config.yaml` is gitignored — it is generated per machine and full of absolute paths.

## Per-solution recipes

Use `work_dir: ./out_nocommit` throughout: `*_nocommit*` is gitignored, so artifacts and
root-owned container output stay out of `git status`.

### `face_obfuscation` — start here

The cheapest target: CPU only, one input, no prep beyond a weights fetch. The best first proof
that the framework path works end to end. Build with the **CPU** `Dockerfile`; GPU demand 0.

`load_config` raises `ValueError("config must set 'input_video'")` — there is no bundled default —
and `output_video` **must end in `.avi`** (`VideofileWriter` supports nothing else).

```yaml
# solutions/face_obfuscation/config.yaml
work_dir: ./out_nocommit
input_video: /home/jadiel/workspace/videoflow-contrib/solutions/offside/data_nocommit/cam0.mp4
output_video: blurred_video.avi
fps: 30
device: cpu
flow_type: batch
detector: {architecture: ssd-mobilenetv2, dataset: faces, num_classes: 1, min_score_threshold: 0.2}
tracker: {max_age: 12, min_hits: 0}
blur: {expand: 0.20, kernel: 23, sigma: 30}
```

The offside clips are soccer wide shots, so the detector will find few or no faces — that still
exercises the whole pipeline and still writes the output. For a semantically meaningful run, use
human_tracking's `people_walking.mp4` sample instead, once its prep hook has downloaded it.

Success artifact: `out_nocommit/blurred_video.avi`, non-empty, fresh mtime.

### `human_tracking`

Needs **no external footage** — an empty `input_video` makes `prepare.py` download the
`people_walking.mp4` sample into `work_dir` (deliberately there rather than the model cache,
because that path is baked into the reader's params and `work_dir` is the same-path mount). It
does need network for that sample plus the encoder weights; detectron2 pose weights are fetched by
the model zoo on first use **inside the worker**, which is an unwarmed runtime dependency and a
plausible blocker. Build with the **CPU** `Dockerfile`; GPU demand 0.

```yaml
# solutions/human_tracking/config.yaml
work_dir: ./out_nocommit
input_video: ''
output_video: annotated_video.avi
device: cpu
flow_type: batch
pose: {architecture: R50_FPN_3x}
encoder: {batch_size: 32}
tracker: {min_height: 0, max_cosine_distance: 0.2, nn_budget: null}
```

Success artifact: `out_nocommit/annotated_video.avi`, non-empty, fresh mtime.

### `offside` — the GPU one, and the most likely to fail

Three cameras, a multi-parent join, and the only GPU demand: **3** — one detector per camera, with
tracker and pose on CPU — against the cluster's 4, so `exclusive` works. Build with
`gpu.Dockerfile`.

**`work_dir` must be `./out_nocommit`.** `prepare.py` runs `weights → sync_offsets → calibrate →
fit_teams`, skipping any step whose output already exists, and **hard-exits telling you to run a
manual click-UI on a machine with a display** if automatic calibration fails. `out_nocommit/`
already holds `calib/cam{0,1,2}.json`, `teams.json` and `offsets.json`, so pointing `work_dir`
there skips all three. It is load-bearing twice over: `build_flow` also calls `cfg.load_offsets()`
and `cfg.load_teams()` at compile time, and both `open()` unconditionally. Point it at the default
`./out` and you get a phantom human-in-the-loop blocker that is really a config error.

```yaml
# solutions/offside/config.yaml
work_dir: ./out_nocommit
flow_type: batch
cameras:
  cam0: {video: data_nocommit/cam0.mp4}
  cam1: {video: data_nocommit/cam1.mp4}
  cam2: {video: data_nocommit/cam2.mp4}
pitch: {length: 105.0, width: 68.0}
attack_direction: auto
team_names: {0: Reds, 1: Blues}
trim: {start_s: null, end_s: null}
detector: {checkpoint: null, resolution: 1288, conf_ball: 0.15, tile_inference: false}
device: {detector: gpu, tracker: cpu, pose: cpu}
fusion: {ref_fps: 30.0, quorum: 2}
debug_overlays: false
```

Camera paths relative to the config dir are correct and consistent: `common.load_config` joins them
against the config dir and `resolve_mounts` joins the same raw values against the graph dir,
producing identical absolute paths — which is what a same-path mount requires.

Success artifact: `out_nocommit/results/verdicts.json`. The visualizer writes it in `close()`
**unconditionally**, so it appears even on a clip with zero offside events. The per-verdict files
(`verdict_N.json`, `still_N_*.png`, `clip_N_*.avi`) only appear when a verdict fires — **their
absence is not a failure.**

`prepare.py` also runs `download_weights.py` (RF-DETR + RTMW pose) relying on the `get_file` cache;
the first run pays the download and a dead mirror is an infra blocker.

### Summary

| Solution | Dockerfile | GPU | Success artifact |
|---|---|---|---|
| `face_obfuscation` | `Dockerfile` (CPU) | 0 | `out_nocommit/blurred_video.avi` |
| `human_tracking` | `Dockerfile` (CPU) | 0 | `out_nocommit/annotated_video.avi` |
| `offside` | `gpu.Dockerfile` | 3 | `out_nocommit/results/verdicts.json` |

## Deploy

```bash
cd /home/jadiel/workspace/videoflow-contrib/solutions/offside
videoflow deploy offside.py \
    --no-build --image videoflow-offside:r1 \
    --config config.yaml --non-interactive --no-prepare \
    --namespace videoflow \
    --flow-id offside --run-id offside-a1 \
    --gpu-runtime-class nvidia --keep-infra
```

| Flag | Why |
|---|---|
| `--no-build --image <ref>:rN` | Builds are yours; dodges the GPU-Dockerfile guess and the `:latest` pull policy. |
| `--config config.yaml` | Feeds mount resolution and `prepare.py`. The file must *also* be at `<solution>/config.yaml`. |
| `--non-interactive` | No TTY. Gives a useful `SystemExit` listing required inputs when the config is missing. |
| `--namespace videoflow` | Isolates the auto-provisioned dev NATS+Redis; makes cleanup one command. |
| `--flow-id` + `--run-id` | Deterministic, and **required** by `teardown` and by every `-l videoflow.io/run-id=` selector. Use DNS-1123-safe hyphens, never `human_tracking`. Increment the run-id every attempt; never reuse one. |
| `--gpu-runtime-class nvidia` | Mandatory; the preflight only warns. Harmless on CPU solutions. |
| `--keep-infra` | Amortises NATS+Redis across runs instead of paying recreation each time. |
| *(no `--keep` by default)* | Deploy already dumps failed nodes' logs before teardown, and auto-teardown frees the GPUs. A kept offside run holds 3 of 4 and makes the next attempt go Pending. Add `--keep` only for a deliberate diagnostic re-run, then tear down immediately. |

Wrap it in `timeout` (say 1800s for offside, 900s for the CPU solutions). A pod stuck in
`ImagePullBackOff` is *not* "Unschedulable", so the 60s watchdog never fires and the wait can hang
indefinitely. Treat a timeout as a triage signal, not a crash.

Long builds and long BATCH runs exceed the foreground command timeout — run them in the background
with output tee'd to a log, then poll.

**Success is a conjunction:**

```bash
grep -q 'Flow offside completed\.' deploy.log                          # 1. the completion line
kubectl get jobs -n videoflow -l videoflow.io/run-id=offside-a1        # 2. nothing failed
find solutions/offside/out_nocommit/results -name verdicts.json -newermt '-30 minutes'   # 3. FRESH artifact
```

Point 3 needs the mtime guard specifically: prep and compile run as root in-image, so a previous
run's artifact is root-owned and cannot be deleted without sudo. Existence alone is not evidence.

## Observing a run

There is **no `videoflow status` and no `videoflow logs`.** Observation is raw `kubectl` plus the
NATS CLI; the only built-in capture is the automatic dump when a BATCH node fails.

```bash
kubectl get pods -n videoflow -l videoflow.io/run-id=<run> -o wide
kubectl logs -n videoflow -l videoflow.io/run-id=<run> --tail=200 --all-containers --prefix
kubectl describe pod <pod> -n videoflow            # names Pending / ImagePull reasons directly
videoflow debug decode --dlq --flow-id <flow> --run-id <run> --nats <url> --limit 20
```

Every resource carries `videoflow.io/flow-id`, `videoflow.io/run-id` and `videoflow.io/node`, so
those selectors scope cleanly to one run.

Health, on container port 8080: `/readyz` returns 200 only after the node's `open()` returns — a
pod stuck not-ready means `open()` is hanging or failing. `/healthz` goes 503 after 60s without a
beat. `/metrics` carries `videoflow_messages_{published,received,processed,failed}_total{node=…}`,
which is how you tell a flow that is working from one that is merely up.

## Teardown

```bash
videoflow teardown --flow-id offside --run-id offside-a1 \
    --nats <url> --namespace videoflow --gpu-mode exclusive

# once, at the very end:
videoflow teardown --flow-id offside --run-id offside-aN \
    --nats <url> --namespace videoflow --infra
```

`--flow-id`, `--run-id` and `--nats` are all required; `--infra` additionally requires
`--namespace`. **Capture the NATS URL from the teardown hint deploy prints** rather than guessing
it. Carry `--gpu-mode` — teardown is the only place a GPU strategy's `cleanup()` runs on the
REALTIME path.

Always tear a run down before advancing to the next run-id, and always before switching solutions
if the previous one held GPUs. Teardown is run-scoped, so it won't disturb a concurrent run — but
**omitting `--run-id` from a manual label delete removes every run of the flow.** A pre-existing
broker is reused and never deleted, by design in both `infra.py` and `localinfra.py`.
