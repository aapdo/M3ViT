## Git Rules

### Commit
- commit message에 "Claude", "AI", "Co-authored" 등 AI 관련 문구를 절대 포함하지 마라
- commit author는 항상 기존 git config의 user.name, user.email을 사용해라
- commit message는 일반적인 컨벤션(feat:, fix:, chore:, refactor: 등)만 사용해라

### Branch
- branch 이름에 "claude", "ai", "auto" 등 AI 관련 문구를 절대 포함하지 마라
- branch 이름은 컨벤션을 따라라: `<type>/<간결한-설명>` (예: feat/token-moe-shareability, fix/imagenet-label-mapping, refactor/models-directory)
- branch 이름만 보고 어떤 작업인지 알 수 있도록 의미 있는 이름을 사용해라
- type: feat, fix, refactor, chore, docs, test 등
