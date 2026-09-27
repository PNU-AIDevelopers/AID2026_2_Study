# AI Researchers (AIR) — Paper Review Handbook

## Overview

- Four members (team **AIR**, branch `Research`), fall semester 2026. No roles — everyone takes part the same way.
- Field: **AI and related areas**. Each member picks papers based on their own interests.
- Goals (from [README](../README.md)):
  1. Every member reads at least four papers. Presenting one paper per review covers this.
  2. Build the AI knowledge and research skills needed to write a short paper.

---

## Schedule

- Weeks alternate: in a **study week** you read on your own, and in a **review week** the group meets offline on **Wednesday at 6 pm**.
- There's no fixed curriculum. Every review follows the same format.

| Week | Date | Type |
| --- | --- | --- |
| 1 | Sep 23 | Study (Chuseok) |
| 2 | Sep 30 | **Review 1** |
| — | Oct 7 / 14 / 21 | Paused for midterms |
| 3 | Oct 28 | Study |
| 4 | Nov 4 | **Review 2** |
| 5 | Nov 11 | Study |
| 6 | Nov 18 | **Review 3** |
| 7 | Nov 25 | Study |
| 8 | Dec 2 | **Review 4** (pre-exam) |

- **Weekly option:** decide at the end of Review 1. If the group isn't satisfied, reviews move to **every Wednesday** from Oct 28 (Oct 28 – Dec 2). The October pre-exam Wednesdays can be used too if everyone agrees. Study weeks then drop out, and each week's folder becomes a review week. Update the README plan table to match.

---

## Papers

- **One paper per person per review**, chosen freely. No duplicates, and papers don't need to share a theme.
- **Post your paper by the Sunday before the review** (the Friday before, in weekly mode).
- **Where to find papers:**
  - Future directions that came up at the last review.
  - Recent venues (NeurIPS / ICML / ICLR / CVPR / ACL, etc.).
  - Whatever you need for your own work or short paper.
  - Social media — but only with a second signal, such as an accepted venue, a strong author track record, or code that runs.
- **The paper queue** lives in [`papers/`](./papers/). Everyone keeps one file listing the papers they want read (link, one line on why) and tags a paper `#reviewed` once the group has covered it. The [website](https://cho104.github.io/AIResearchers/) shows every list.

### OpenReview rejected-paper review (at least once)

- Agree on a date in advance. For that review, each person brings **one paper rejected on OpenReview** (e.g. an ICLR or NeurIPS submission with public reviews).
- Present the paper briefly, then **why it was rejected**, using the reviews, the rebuttal, and the meta-review.
- Say whether you agree with the decision. Would you have accepted it, and what would have saved it?
- This is where critique and paper-writing skills meet: you see exactly which problems reviewers punish.

---

## Study weeks

- Read your own paper deeply and prepare a presentation PDF (problem → key idea → evidence → limits).
- Skim the other three papers (title, abstract, figures) and prepare one question for each. Using AI to help is fine.
- Add your notes to `weekXX/README.md` (from `WEEK_README_TEMPLATE.md`) and upload your study PDF.

---

## Review sessions

- **No time limits.** Each paper takes as long as it needs. Take breaks when needed.
- **For each paper:**
  1. **Presentation.** Problem → key idea → evidence → limits, not a section-by-section walkthrough. You should be able to state the central claim in one sentence and sketch the method without slides.
  2. **Questions.** Everyone who didn't present asks **at least one question**. Basic, critical, or prepared with AI — all count.
  3. **What's missing?** Claims without evidence, weak baselines and ablations, reproducibility gaps (code, data, seeds, compute), and limitations the authors didn't admit.
  4. **Future directions.** What would the next paper do? Could any of it become someone's short paper?
- **Wrap-up:** which future direction is most worth following up, and does it change anything we're doing?
- **Tone:** critique as if the authors are in the room. "They didn't do X" is only a flaw if you can say why X was needed.

---

## Recording the meetings

Get everyone's consent before recording, and keep raw recordings out of the repo. INSTRUCTION.md asks for no large files and no personal information.

- **Voice transcript → summary:**
  - Transcribe each meeting with a speech-to-text tool: e.g. Clova Note (handles Korean well), or Whisper run locally.
  - Have an AI turn the transcript into a summary covering each paper's key points, the questions asked, what's missing, future directions, and the verdict.
  - Check the summary, then put it in that week's `weekXX/README.md`. Upload only the summary, not the raw transcript.
- **Video:**
  - Record every review.
  - Keep the videos in a shared drive, not in the repo.
  - At the end of the semester, pick the best presentation together and post it, with the presenter's consent. Link it from the README.

---

## Records and PRs

Follows [INSTRUCTION.md](./INSTRUCTION.md). All activity lives on `main`: members fork this repo and open PRs against `main`.

- **Weekly folders:** `week01/` … `week08/`, each with a `README.md`. The midterm pause gets no folder.
  - Study week: each member's notes and study PDF.
  - Review week: the meeting summary, presentation PDFs, and the 회고.
- **PR:** one per week, titled `[AIR] N주차 활동 인증`, using the PR body format in INSTRUCTION.md.
- **Paper PDFs:** link them instead of uploading them.

---

## If something goes wrong

- **Someone is absent:** run the review with three papers. The absent member presents at the next review. If three people are absent, reschedule.
- **Someone didn't read a paper:** the one-question duty still stands, and an AI-prepared question counts.
