# Этап 2: EXECUTE

По результату этапа 1 записать задачи, сохранить заметки, обновить проекты, построить связи.

## Вход

- Результат этапа 1 (или `.session/capture.json`)
- `projects/_index.md` и `projects/{slug}/status.md` упомянутых проектов

## Задача

### 1. Записать задачи

Для каждой записи `classification: "task"`:

1. Проверь дубликаты: поищи похожие открытые `- [ ]` в `daily/` за 14 дней и
   в `projects/*/status.md`. Если есть, не создавай, пометь запись
   `<!-- ✓ processed: task (duplicate) -->`.
2. Проверь загрузку (см. `references/rules.md`): если на день уже 3+ задачи,
   сдвинь срок на ближайший менее загруженный день (кроме p1).
3. Добавь чек-лист в раздел `## Задачи` daily-файла, а если задача про
   конкретный проект, ещё и в `projects/{slug}/status.md`:

```markdown
## Задачи
- [ ] {task} (due: YYYY-MM-DD, p2) → [[projects/{slug}/status|Название]]
```

### 2. Сохранить заметки

Для idea / reflection / learning / project:

- Файл `thoughts/{category}/YYYY-MM-DD-slug.md`
- Frontmatter по шаблону autograph: `type: note`, `description` (фраза для
  поиска, до ~150 символов), `tags` (2-5), `status`, `created`, `source`
- Раздел `## Related` с типизированными связями:
  ```markdown
  ## Related
  - [[projects/proekt-a/status|Проект А]] — context: обсуждали на планёрке
  ```

### 3. Обновить проекты

- Новость по проекту: строка в разделе «История» `projects/{slug}/status.md`.
- Существенная смена состояния: сначала копия в
  `projects/{slug}/archive/YYYY-MM-DD.md`, потом новая версия `status.md`.

### 4. Построить связи

Для всех созданных и обновлённых файлов (подробно: `references/links.md`):
- найди связанные заметки в vault;
- добавь wiki-ссылки с поясняющей фразой;
- обнови `related: []` во frontmatter и нужный файл в `MOC/`.

### 5. Пометить обработанное

После каждой записи в daily добавь маркер `<!-- ✓ processed: {category} -->`.
Сам текст записи не меняй.

## Формат результата

Только валидный JSON (при желании сохрани в `.session/execute.json`):

```json
{
  "tasks_created": [
    {"content": "Отправить финансовому отделу черновик бюджета Проекта А", "priority": 2, "due": "2026-03-13", "file": "daily/2026-03-12.md"}
  ],
  "tasks_skipped_duplicates": 0,
  "thoughts_saved": [
    {"path": "thoughts/learnings/2026-03-12-weekly-review-calendar-first.md", "title": "Недельный обзор быстрее начинать с календаря", "category": "learnings"}
  ],
  "projects_updated": [
    {"path": "projects/proekt-a/status.md", "change": "Срок согласования бюджета: пятница"}
  ],
  "links_created": [
    {"from": "thoughts/learnings/2026-03-12-weekly-review-calendar-first.md", "to": "goals/3-weekly.md", "context": "supports: недельный фокус"}
  ],
  "open_tasks": {"active": 7, "overdue": 1, "today": 2},
  "workload": {"mon": 3, "tue": 2, "wed": 4, "thu": 1, "fri": 2, "sat": 0, "sun": 0},
  "observations": []
}
```
