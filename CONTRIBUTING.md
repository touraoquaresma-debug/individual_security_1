# Guia de Contribuição 🤝

## Como Contribuir

Agradecemos seu interesse em contribuir com o projeto! Aqui estão as diretrizes para ajudar você nesse processo.

### 1. Preparando o Ambiente

1. Faça um fork do repositório
2. Clone seu fork:
```bash
git clone https://github.com/seu-usuario/YOLO-Elderly-Pose-Detection-Monitoring.git
cd YOLO-Elderly-Pose-Detection-Monitoring
```
3. Instale as dependências de desenvolvimento:
```bash
pip install -e ".[dev]"
```

### 2. Desenvolvimento

1. Crie uma branch para sua feature:
```bash
git checkout -b feature/nome-da-feature
```

2. Mantenha seu código limpo:
- Use `task format` para formatar o código
- Use `task lint` para verificar problemas
- Execute `task test` para rodar os testes

3. Commits:
- Use mensagens claras e em português
- Prefixe com o tipo: `feat:`, `fix:`, `docs:`, `test:`, etc.

### 3. Enviando Contribuições

1. Atualize sua branch:
```bash
git fetch origin
git rebase origin/main
```

2. Envie suas alterações:
```bash
git push origin feature/nome-da-feature
```

3. Abra um Pull Request:
- Descreva claramente as mudanças
- Inclua evidências de testes
- Referencie issues relacionadas

### 4. Diretrizes de Código

- Siga PEP 8
- Documente funções e classes
- Mantenha cobertura de testes
- Use type hints

### 5. Issues

- Verifique issues existentes antes de criar novas
- Use os templates disponíveis
- Seja claro e específico
- Inclua passos para reprodução de bugs

### 6. Processo de Review

- Cada PR precisa de pelo menos 1 aprovação
- CIs devem passar
- Mudanças solicitadas devem ser implementadas

## Código de Conduta

- Seja respeitoso
- Aceite feedback construtivo
- Foque em colaboração
- Mantenha discussões profissionais

## Dúvidas?

Abra uma issue ou entre em contato através do GitHub.

---

🙏 Obrigado por contribuir!
