# How to resolve a rebase conflict (quick steps)

1. Check current rebase status:
   git status
   git rev-parse --abbrev-ref HEAD

2. See conflicted files:
   git diff --name-only --diff-filter=U

3. Edit each conflicted file and remove conflict markers (<<<<<<<, =======, >>>>>>>), choose correct code, save.

4. Stage fixed files:
   git add <file1> <file2> ...

5. Continue rebase:
   git rebase --continue

6. If something is irrecoverable, abort:
   git rebase --abort

7. After successful rebase, push (use force-with-lease when rewriting history):
   git push --force-with-lease origin main

Notes:
- If there are multiple commits with conflicts, repeat steps 2–5 for each.
- Use a diff/merge tool (e.g., VS Code, Beyond Compare) if helpful.
