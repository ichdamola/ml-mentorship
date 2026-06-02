# Week 03: The Python ML Stack

## 🎯 What you'll learn

Fluency in the day-to-day Python tools you'll use every week for the rest of the curriculum. Most of what slows people down isn't the math — it's not knowing the difference between `np.array` and `np.matrix`, or that `df.iloc[0]` returns a Series.

By the end of this week you'll be able to:

- Write fluent numpy: broadcasting, axis-aware reductions, fancy indexing, vectorization
- Use pandas for tabular data: groupby, merge, pivot, time series
- Plot with matplotlib at the level of "this would belong in a paper"
- Set up a reproducible project with `uv`, `pyproject.toml`, and `requirements.lock`
- Use Jupyter without it becoming a junk drawer (kernel restarts, autoreload, magics)

## 🧰 Lab setup

```bash
uv add numpy pandas matplotlib jupyter ipykernel scikit-learn
```

## ✅ Your job

1. Read [theory.md](theory.md) — the broadcasting model and why it's the killer feature of numpy
2. Work through [lab.md](lab.md) — vectorize a Python loop and measure the 100× speedup
3. Stretch: read your own old loopy code and find one place to vectorize it

## 📚 Required reading

| Resource | Why | Time |
|---|---|---|
| [numpy broadcasting docs](https://numpy.org/doc/stable/user/basics.broadcasting.html) | Required for everything that follows | 30 min |
| [pandas 10-minute intro](https://pandas.pydata.org/docs/user_guide/10min.html) | Concise tour | 30 min |
| [Jake VanderPlas — Python Data Science Handbook (numpy + pandas chapters)](https://jakevdp.github.io/PythonDataScienceHandbook/) | The deepest free resource | 90 min |

## 💡 What you should already know

- Python (loops, functions, classes)
- Weeks 01-02

---

> 🚧 **Scaffolded.** Full theory + lab content lands in the next pass.

**Next**: [Week 04: Classical ML →](../week-04-classical-ml/readme.md)
