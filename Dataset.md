## Dataset Analysis

The dataset contains metadata for 131 live-streaming anchors.

* **Gender:** 120 female anchors and 11 male anchors. The dataset is highly imbalanced, so the recommendation system may naturally recommend more female anchors.
* **Personality:** The most common personality is `純真可愛型`. Multi-label values such as `純真可愛型、幽默搞笑型` are split into separate labels before analysis.
* **Talents:** Singing is the dominant talent, followed by chatting, dancing, food-stream interaction, and interactive games.
* **Featured topics:** Most anchors focus on casual conversation, music companionship, and light interaction.
* **Streaming style:** The most common styles are friendly, highly interactive, and companion-like.

Because several columns contain multiple labels in one cell, Power Query is used to split them into separate rows for accurate counting and visualization.

Power BI is used to explore the dataset, compare anchor characteristics, and monitor recommendation results. The recommendation logic itself is implemented in Python.
