Носов Артём Иванович	М24-534	Parameters, Settings, and Performance of Code LLMs: An Comparative Study (Параметры, настройки и производительность БЯМ для работы с кодом: сравнительное исследование)	Парсинг HuggingFace через API	выявить ключевые архитектурные и гиперпараметрические признаки кодовых больших языковых моделей, статистически значимо влияющие на их производительность, а также обнаружить аномальные экземпляры, отклоняющиеся от общей зависимости «признаки → эффективность»	SHAP, Partial Dependence Plots, Self-Organizing Maps, Isolation Forest, LOF (Local Outlier Factor), Boruta, Permutation Feature Importance, Gaussian Process Regression, Variational Autoencoders (VAE), Spectral Clustering	Предварительные выводы показывают, что производительность кодовых БЯМ в наибольшей степени зависит от количества параметров, типа предобучения и длины контекста, при этом отдельные модели демонстрируют аномально высокую или низкую эффективность при схожих характеристиках, что указывает на потенциальное влияние скрытых факторов — таких как качество обучающего корпуса или особенности дообучения.	"Какие именно признаки кодовых БЯМ будут анализироваться?
По каким метрикам оценивается их производительность?
Все ли выбранные методы нужны при ограниченном числе моделей?"	"1 ['likes',
 'downloads',
 'created_at',
 'url',
 'tags',
 'Pipeline Tag',
 'Architectural Type',
 'Real Architecture',
 'Parameters (Parsed)',
 'Version (Parsed)',
 'Quantization',
 'Is Instruct/Chat (Parsed)',
 'Version (Parsed) (float)'] 2 Производительность оценивается по стандартным NLP-бенчмаркам (ARC, GSM8K, HellaSwag, MMLU, TruthfulQA, Winogrande), метрикам UGI-Leaderboard (Coding, UGI, W/10) и специализированным метрикам (Grounding, Instruction Following, Planning, Reasoning, Safety). 3 Методы отобраны с учётом задачи: часть — для интерпретации (SHAP, Boruta), часть — для поиска аномалий (Isolation Forest, LOF); при необходимости набор будет сокращён."