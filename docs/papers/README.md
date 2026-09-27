# Papers

Everyone keeps one file here listing the papers they want the group to read. The [website](https://cho104.github.io/AIResearchers/) shows every list, and it updates a few minutes after a PR is merged into `main`.

## Add your list

1. Fork this repo.
2. Create `docs/papers/<your-github-id>.md`:

   ```markdown
   # 이치오

   - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — cheap fine-tuning baseline for my short paper #peft
   - https://arxiv.org/abs/1706.03762
   ```

3. Open a PR against `main`.

To add more papers later, add lines to your file and open another PR.

## Format

- The `# heading` is your name on the site.
- Each paper is one line starting with `- `. Write `[Title](link)`, or paste the bare link.
- Anything after the link is your one line on why.
- Words starting with `#` are tags, like `#llm`, or `#rejected` for the OpenReview review.
- Papers show as **TBD** until you add `#reviewed` to the line.
- If two people add the same paper, the site shows it once with both names.
- Other headings and indented lines stay in your file but don't show on the site.

## JSON instead

`docs/papers/<your-github-id>.json` works too. Only `link` is required, and `status` is `"tbd"` unless you set it to `"reviewed"`:

```json
{
  "name": "이치오",
  "papers": [
    { "link": "https://arxiv.org/abs/2106.09685", "title": "LoRA", "why": "cheap fine-tuning baseline", "tags": ["peft"], "status": "reviewed" },
    { "link": "https://arxiv.org/abs/1706.03762" }
  ]
}
```

## Preview before opening a PR

From the repo root, run `python3 -m http.server -d docs`, then open http://localhost:8000. If a file can't be read, the page says which one and why.
