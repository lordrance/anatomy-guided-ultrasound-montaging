# Scope of the licence, and third-party material

The MIT licence in [`LICENSE`](LICENSE) covers **the source code in this
repository only**. Everything below belongs to someone else and is not
relicensed here.

## Datasets — not redistributed

No dataset is included in this repository. The download scripts fetch data from
its official source at run time, and each dataset stays under its own terms:

| dataset | source | licence |
|---|---|---|
| PTB-XL v1.0.3 | PhysioNet | CC BY 4.0 |
| Shoulder 3D Ultrasound Mosaicking Dataset | Zenodo `10.5281/zenodo.20283247` | CC BY 4.0 |

Both were collected from human participants under their own ethics approvals,
which are documented by their publishers. Anyone reusing them is bound by those
terms, not by this repository's licence.

## Papers

The work here is inspired by, and cites, published research. No text, figure,
table or code from those papers is reproduced in this repository; they are
referenced so a reader can find them.

- Zhang Y, Padhee S, Yuhas PT, Roberts CJ, Parthasarathy S. *Adaptive Temporal
  Mixture of Experts for Predicting Stiffness Metrics From the Ocular Response
  Analyzer and Identifying Keratoconus.* American Journal of Ophthalmology,
  2026;286:196–210. doi:10.1016/j.ajo.2026.02.028
- Zhang Y, et al. *Automated Montaging of Ultrasound Scans for Extended-Range
  Imaging of the Posterior Eye.* Research Square preprint, 2026.
  doi:10.21203/rs.3.rs-8835197/v1
- Sewify A, Steffens M, Perrier N, Antico M, Lavaill M, Edwards C, Pivonka P,
  Fontanarosa D. *Comprehensive reconstruction of the musculoskeletal anatomy in
  the shoulder using a hybrid 3D ultrasound mosaicking workflow: a pilot study.*
  PLOS ONE, 2026. doi:10.1371/journal.pone.0347231
- Wagner P, Strodthoff N, Bousseljot R-D, et al. *PTB-XL, a large publicly
  available electrocardiography dataset.* Scientific Data, 2020;7:154.

## Software

Built on NumPy, SciPy, scikit-learn, scikit-image, Matplotlib, PyTorch,
openpyxl and pytest, each used as a dependency under its own licence. No
third-party source code is vendored into this repository.
