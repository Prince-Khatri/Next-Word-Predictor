const input = document.getElementById('main-input');
const ghost = document.getElementById('ghost-display');
let nextWord = "";

async function getNextWord() {
    const currentText = input.value;
    
    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: currentText.trim() })
        });
        const data = await res.json();
        nextWord = data.prediction || "";
        showGhost();
    } catch (err) {
        console.log("Prediction failed", err);
    }
}

function showGhost() {
    const currentVal = input.value;
    ghost.innerHTML = currentVal + '<span class="prediction-text">' + nextWord + '</span>';
}

input.addEventListener('input', () => {
    nextWord = "";
    showGhost();
});

input.addEventListener('keydown', (e) => {
    if (e.key === ' ') {
        setTimeout(getNextWord, 10);
    }

    if (e.key === 'Tab' && nextWord !== "") {
        e.preventDefault();
        input.value = input.value + nextWord;
        nextWord = "";
        showGhost();
    }
});

input.addEventListener('scroll', () => {
    ghost.scrollTop = input.scrollTop;
});