class TextVisualizer {
    constructor(leftPanel, rightPanel, arrowsSvg, addedWordsElement, wordCountScoreElement) {
        this.leftPanel = leftPanel;
        this.rightPanel = rightPanel;
        this.arrowsSvg = arrowsSvg;
        this.addedWordsElement = addedWordsElement;
        this.wordCountScoreElement = wordCountScoreElement;
    }

    clearVisualization() {
        this.leftPanel.innerHTML = '';
        this.rightPanel.innerHTML = '';
        this.arrowsSvg.innerHTML = '';
    }

    updateStats(comparison) {
        this.addedWordsElement.textContent = comparison.added_words;
        this.wordCountScoreElement.textContent = comparison.word_count_score;
    }

    visualizeLeftText(comparison) {
        comparison.left_tokens.forEach((token, index) => {
            const span = document.createElement('span');
            span.textContent = token.text;
            span.dataset.index = index;
            span.classList.add('token');
            
            // Check if this token is part of moved text or just matched/removed
            let matchType = 'removed';
            for (const match of comparison.matches) {
                if (index >= match.left_start && index < match.left_start + match.length) {
                    // Check if this match represents moved text by comparing positions
                    const leftPos = match.left_start;
                    const rightPos = match.right_start;
                    matchType = (leftPos !== rightPos) ? 'moved' : 'matched';
                    break;
                }
            }
            span.classList.add(matchType);

            // Preserve whitespace exactly
            if (token.text.includes(' ') || token.text.includes('\n') || token.text.includes('\t')) {
                span.style.whiteSpace = 'pre';
            }
            
            this.leftPanel.appendChild(span);
        });
    }

    visualizeRightText(comparison) {
        comparison.right_tokens.forEach((token, index) => {
            const span = document.createElement('span');
            span.textContent = token.text;
            span.dataset.index = index;
            span.classList.add('token');
            
            // Check if this token is part of moved text or just matched/added
            let matchType = 'added';
            for (const match of comparison.matches) {
                if (index >= match.right_start && index < match.right_start + match.length) {
                    // Check if this match represents moved text by comparing positions
                    const leftPos = match.left_start;
                    const rightPos = match.right_start;
                    matchType = (leftPos !== rightPos) ? 'moved' : 'matched';
                    break;
                }
            }
            span.classList.add(matchType);

            // Preserve whitespace exactly
            if (token.text.includes(' ') || token.text.includes('\n') || token.text.includes('\t')) {
                span.style.whiteSpace = 'pre';
            }
            
            this.rightPanel.appendChild(span);
        });
    }

    drawMatchArrows(matches) {
        // Clear existing arrows
        this.arrowsSvg.innerHTML = '';
        // For now, we'll skip drawing arrows as they're not crucial
        // and we need to fix the basic visualization first
    }
}
