import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "ai-infrastructure/architecture"
C = {"ml": "#DCEBFA", "store": "#FDEBD3", "serve": "#DFF3E4", "ops": "#EFE4F7", "ext": "#F1F1F1"}

def draw(title, boxes, arrows, groups, fname, size=(14, 8)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
    for (x, y, w, h, label) in groups:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="none", ec="#888", ls="--", lw=1.2))
        ax.text(x + 0.8, y + h - 0.6, label, fontsize=10, color="#555", va="top", style="italic")
    pos = {}
    for key, (x, y, w, h, text, kind) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                    fc=C[kind], ec="#333", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
        pos[key] = (x, y, w, h)
    def anchor(k, side):
        x, y, w, h = pos[k]
        return {"r": (x + w + 0.3, y + h / 2), "l": (x - 0.3, y + h / 2),
                "t": (x + w / 2, y + h + 0.3), "b": (x + w / 2, y - 0.3)}[side]
    for a, sa, b, sb, label in arrows:
        p1, p2 = anchor(a, sa), anchor(b, sb)
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#333"))
        if label:
            ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.8, label, fontsize=7.5, ha="center",
                    color="#333", bbox=dict(fc="white", ec="none", pad=0.5))
    ax.set_title(title, fontsize=13, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=C[k], ec="#333") for k in ("ml", "store", "serve", "ops")]
    ax.legend(handles, ["ML code / pipeline", "storage & registry", "serving", "CI/CD & monitoring"],
              loc="lower right", fontsize=8, frameon=False, ncol=4)
    fig.savefig(f"{OUT}/{fname}", dpi=160, bbox_inches="tight")

# ---------------- current ----------------
boxes = {
 "data": (2, 44, 14, 7, "GBG500.parquet\n+ labels.csv\n(local, git-ignored)", "store"),
 "prep": (20, 44, 14, 7, "trajclf.data\nduplicates avg, NaN\n30 stats | sequence", "ml"),
 "train": (38, 44, 14, 7, "trajclf.train\nPyTorch MLP / CNN\nCPU laptop or Colab T4", "ml"),
 "mlflow": (58, 44, 15, 7, "MLflow server\ntracking + Registry\n(sqlite)", "store"),
 "minio": (78, 44, 15, 7, "MinIO (S3)\nbucket mlflow/\nbundle: .pt .onnx", "store"),
 "api": (58, 24, 15, 8, "FastAPI pods (x2)\n/predict /metrics\nTorchScript, 1 thread", "serve"),
 "web": (78, 24, 15, 8, "Streamlit web app", "serve"),
 "user": (38, 24, 14, 8, "User / load test\n(HTTP JSON ride)", "ext"),
 "prom": (58, 6, 15, 7, "Prometheus\nscrape 5 s", "ops"),
 "graf": (78, 6, 15, 7, "Grafana\nAPI + model dashboard", "ops"),
 "gh": (2, 24, 16, 8, "GitHub Actions\nCI: ruff + pytest\nbuild & push Docker Hub", "ops"),
 "runner": (2, 6, 16, 8, "Self-hosted runner\nCD: kubectl apply,\nrollout / undo", "ops"),
}
arrows = [("data","r","prep","l",""),("prep","r","train","l","splits.npz"),("train","r","mlflow","l","params, metrics"),
 ("mlflow","r","minio","l","artifacts"),("minio","b","api","t","download @champion"),
 ("user","r","api","l","POST /predict"),("web","l","api","r",""),("api","b","prom","t","/metrics"),
 ("prom","r","graf","l",""),("gh","b","runner","t","images :sha"),("runner","r","prom","l","deploys to cluster")]
groups = [(55, 2, 42, 52, "Docker Desktop Kubernetes (namespace trajclf)  —  docker compose in dev")]
draw("Current system — single machine (local Kubernetes / Docker Compose)", boxes, arrows, groups, "current-system.png")

# ---------------- proposed ----------------
boxes = {
 "client": (2, 40, 13, 8, "Clients\n(app / batch)", "ext"),
 "lb": (19, 40, 14, 8, "Cloud Load Balancer\n+ Ingress (HTTPS)", "serve"),
 "api": (38, 36, 18, 12, "API Deployment\nONNX Runtime or TorchScript\n(fastest measured)\nHPA 2→6 pods, CPU 60 %\n0.25 vCPU / 0.5 GiB each", "serve"),
 "web": (38, 20, 18, 7, "Web app (1 pod)", "serve"),
 "gcs": (61, 44, 15, 7, "GCS bucket\nmodel bundles + data", "store"),
 "mlflow": (81, 44, 16, 7, "MLflow (1 pod)\nbackend Cloud SQL\nalias @champion", "store"),
 "train": (81, 27, 16, 8, "Retrain Job / CronJob\n1 CPU pod (or 1 T4)\nno DDP needed", "ml"),
 "mon": (61, 20, 15, 9, "Prometheus + Grafana\nlatency p95, errors,\nclass mix, confidence", "ops"),
 "alert": (81, 8, 16, 8, "Alerts\np95 > SLO, 5xx > 1 %\ndrift → retrain", "ops"),
 "cicd": (2, 12, 15, 9, "GitHub Actions\nCI tests → image :sha\nCD rolling update\nsmoke test → undo", "ops"),
 "ar": (19, 12, 14, 9, "Artifact Registry\n/ Docker Hub\n(immutable tags)", "store"),
}
arrows = [("client","r","lb","l",""),("lb","r","api","l",""),("gcs","l","api","r","bundle"),
 ("mlflow","l","gcs","r",""),("train","t","mlflow","b","register, promote if F1 ↑"),
 ("api","b","web","t",""),("api","r","mon","t","/metrics"),("mon","r","alert","l",""),
 ("alert","t","train","b","trigger"),("cicd","r","ar","l",""),("ar","t","api","b","rollout")]
groups = [(35, 4, 64, 51, "GKE Autopilot cluster, europe-west1 (proposed — not deployed / not tested)")]
draw("Proposed system — scalable, cost-aware serving on GKE Autopilot", boxes, arrows, groups, "proposed-system.png")
print("ok")
