# pitchaero-assessment
Take home assessment for Pitch Aeronautics: Wind Forecast Post-Processing

## Context

This role is mostly ML development, with some data-pipeline and integration work and a bit of AWS. This exercise is a small, realistic slice of that work. We're less interested in a high leaderboard score than in how you frame a messy problem, validate honestly, and communicate your choices.

## The task

Using the data we provide, **train a model that improves HRRR's wind forecasts** at our sensor site.

Put plainly: given HRRR's forecast, can you produce a *corrected* wind forecast that lands closer to what our sensor actually measured? How you frame the supervised problem - what you predict, at what time resolution, what you treat as features vs. ground truth - is part of the exercise. We've intentionally not done that framing for you.

## What we provide

- `sensor_wind.csv` (shipped gzipped) - HRRR forecast fields paired with observations from a WireWarrior sensor over `2025-04-01 to 2026-04-01 (UTC)`.
- `sensor_wind_data_dictionary.md` - the data dictionary for the file above.

About the sensor:
- It reports **every minute.**
- It sits **22.1 meters** above the ground.
- It's located at **42.8783, -115.0089**.
- Wind is reported in **m/s**.

The high-frequency data gives you a lot of room to work with - how you use it is up to you. The file is real data and reflects real-world quirks. Look at it before you model it.

## Ground rules

- **Budget ~2–3 hours of effort.** Please don't exceed 3. If you hit a wall, stop and write up what you'd do next - that's a perfectly good answer and we read it carefully.
- You have **72 hours** to return it.
- **Use any tools you like, including AI assistants** (Claude Code, Copilot, ChatGPT, etc.). We use these every day and want to see how you actually work now - not how you'd work without them.
- You **may** pull in additional outside data if you think it improves the model. Not required, and not doing so won't count against you. We're interested in your reasoning about data either way.

## How to measure yourself

- Report **RMSE and MAE** of your corrected forecast vs. observations on a **held-out test set**.
- Report the same metrics for **raw HRRR** (i.e., the uncorrected forecast as the baseline).
- Headline number: **% RMSE improvement over raw HRRR.** If your model doesn't beat the baseline, tell us - an honest "it didn't beat HRRR and here's why" is more valuable than a number you don't trust.

## Communicating your results

**Communicate and visualize your results in the way you think best.** We want to understand not just your final numbers but where your model helps, where it doesn't, and why. The specific form is up to you - but show us, don't just tell us.

## Looking ahead (please include)

Spend **a few minutes** sketching what you'd do with a longer runway. If you had a few weeks instead of a few hours:

- What would you **prioritize** first, and why?
- Which **additional features or data sources** do you think would move the needle most - especially from a meteorological standpoint?
- What **model types or analyses** would you want to try, and what would you expect each to buy you?

We're looking for your reasoning here as much as your results.

## Deliverables

1. **Code** - a zip or a git link, runnable from a short README. Reproducibility matters more than polish.
2. **A README, roughly one page**, covering:
   - How you framed the problem (what's the target, at what resolution, why) and how you split the data.
   - Your baseline vs. model results.
   - Any data issues you noticed and how you handled them.
   - **Visual evidence of your results, not just metrics.**
   - Your **"looking ahead"** notes (see above).

## What happens next

We'll schedule a **30–45 minute walkthrough** where you talk us through your choices and we ask you to extend or change something live.
