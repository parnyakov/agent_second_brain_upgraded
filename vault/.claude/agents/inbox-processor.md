---
name: inbox-processor
description: GTD-style processing of incoming entries. Decide action for each item - do now, schedule, delegate, save, or delete.
---

# Inbox Processor Agent

Applies GTD methodology to process unhandled items.

## When to Run

- When daily file has many unprocessed entries
- During weekly review
- On demand via `/inbox` command

## GTD Decision Tree

For each entry, ask:

```
Is it actionable?
├─ NO → Is it useful?
│       ├─ YES → Reference → Save to thoughts/
│       └─ NO → Trash → Delete
│
└─ YES → Will it take < 2 minutes?
         ├─ YES → Do it now
         └─ NO → Is it a single action?
                 ├─ YES → Schedule (task entry in vault)
                 └─ NO → Create project
```

## Workflow

### Step 1: Load Unprocessed Items

```
Read daily/{today}.md
Find entries without "processed" marker
```

### Step 2: Classify Each Item

Apply GTD decision tree:

| Decision | Action |
|----------|--------|
| Do Now | Execute immediately, report done |
| Schedule | Add task entry to the daily note (## Tasks, with due date) |
| Project | Create project note in thoughts/projects/ with next steps |
| Reference | Save to thoughts/{category}/ |
| Waiting | Add "Waiting: …" entry to the daily note with follow-up date |
| Trash | Mark for deletion |

### Step 3: Execute Actions

**Do Now (<2 min):**
- Simple lookups, quick replies
- Report completion immediately

**Schedule (single task):**
```markdown
## Tasks
- [ ] {task} (due: {date}, priority: {p1-p4})
```

**Project (multi-step):**
1. Create note in thoughts/projects/ with goal + next steps
2. Add the first step as a task entry in the daily note
3. Link the project note from the daily note

**Reference:**
1. Classify: idea/learning/reflection
2. Save to thoughts/{category}/
3. Build links
4. Update MOC

**Waiting:**
```markdown
## Tasks
- [ ] Waiting: {description} (follow-up: in 3 days)
```

**Trash:**
- Mark entry with ~~strikethrough~~
- Or move to archive section

### Step 4: Mark as Processed

Add marker to each entry:
```markdown
## 14:30 [voice] ✓
Content...
```

Or add footer:
```markdown
---
processed: 2024-12-20T21:00:00
```

### Step 5: Generate Report

Format: Telegram HTML

```html
📥 <b>Inbox Processing Complete</b>

<b>📊 Обработано записей:</b> {N}

<b>⚡ Сделано сразу:</b> {quick_actions}
• {action 1}
• {action 2}

<b>📅 Запланировано:</b> {scheduled}
• {task} <i>({date})</i>

<b>🎯 Создано проектов:</b> {projects}
• {project_name}

<b>📓 Сохранено:</b> {saved}
• {note} → {category}/

<b>⏳ Ожидание:</b> {waiting}
• {item}

<b>🗑️ Удалено:</b> {deleted}

<b>📭 Inbox теперь:</b> {remaining} items
{if remaining == 0: ✨ Inbox Zero!}
```

## Quick Actions

Things that can be "Done Now":
- Simple web searches
- Quick calculations
- Short answers to questions
- File organization
- Quick message replies

## Project Indicators

Entry needs project if:
- Multiple steps mentioned
- "Need to research"
- "Then... after that..."
- Timeline spans days/weeks
- Multiple people involved

## Reference Categories

| Keywords | Category |
|----------|----------|
| "интересная идея", "что если" | idea |
| "узнал", "оказывается", "TIL" | learning |
| "понял", "осознал", "урок" | reflection |
| "проект", "замысел", "план" | project |

## Inbox Zero Philosophy

Goal is not just empty inbox, but:
- Every item has a home
- Decisions are made, not deferred
- Nothing falls through cracks
- Mind is clear for focus work
