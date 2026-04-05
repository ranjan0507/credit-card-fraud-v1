// ----------------------------------------------------
// PREDEFINED DATA PROFILES (V1-V28 PCA Vectors + Time)
// ----------------------------------------------------

// Legitimate profile — realistic non-fraud transaction
// (small PCA variations, moderate values from training data)
const legitData = {
    "Time": 0.0,
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    "V3": 2.53634673796914,
    "V4": 1.37815522427443,
    "V5": -0.338320769942518,
    "V6": 0.462387777762292,
    "V7": 0.239598554061257,
    "V8": 0.0986979012610507,
    "V9": 0.363786969611213,
    "V10": 0.0907941719789316,
    "V11": -0.551599533260813,
    "V12": -0.617800855762348,
    "V13": -0.991389847235408,
    "V14": -0.311169353699879,
    "V15": 1.46817697209427,
    "V16": -0.470400525259478,
    "V17": 0.207971241929242,
    "V18": 0.0257905801985591,
    "V19": 0.403992960255733,
    "V20": 0.251412098239705,
    "V21": -0.018306777944153,
    "V22": 0.277837575558899,
    "V23": -0.110473910188767,
    "V24": 0.0669280749146731,
    "V25": 0.128539358273528,
    "V26": -0.189114843888824,
    "V27": 0.133558376740387,
    "V28": -0.0210530534538215,
    "Amount": 149.62
};

// Fraud profile — REAL fraud case from training data
// (dataset index 235644, closest to the median of all fraud cases, predicts 99.99% fraud probability)
const fraudData = {
    "Time": 148479.0,
    "V1": -1.541678,
    "V2": 3.8468,
    "V3": -7.604114,
    "V4": 3.121459,
    "V5": -1.254924,
    "V6": -2.084875,
    "V7": -2.385027,
    "V8": 1.47114,
    "V9": -2.530507,
    "V10": -5.17566,
    "V11": 1.927186,
    "V12": -6.011155,
    "V13": -1.195601,
    "V14": -6.745561,
    "V15": -0.058091,
    "V16": -4.325132,
    "V17": -5.558067,
    "V18": -1.580531,
    "V19": 0.971906,
    "V20": 0.11476,
    "V21": 1.096405,
    "V22": 1.064222,
    "V23": 0.06537,
    "V24": 0.257209,
    "V25": -0.693654,
    "V26": -0.335702,
    "V27": 0.577052,
    "V28": 0.398348,
    "Amount": 122.68
};

// Validation constants
const AMOUNT_MIN = 0;
const AMOUNT_MAX = 25000;


// ----------------------------------------------------
// STATE & DOM ELEMENTS
// ----------------------------------------------------
let currentScenario = 'legit';
let isDevMode = false;

const txAmountInput = document.getElementById('txAmount');
const jsonPayloadTextarea = document.getElementById('jsonPayload');

// Listen for amount changes to update JSON dynamically
txAmountInput.addEventListener('input', () => {
    clearAmountError();
    updateJSONPayloadView();
});

// Initialize view
setScenario('legit');


// ----------------------------------------------------
// SCENARIO TOGGLE LOGIC
// ----------------------------------------------------
function setScenario(type) {
    currentScenario = type;

    // Update Button UI
    document.getElementById('btnLegit').classList.remove('active');
    document.getElementById('btnFraud').classList.remove('active');

    if (type === 'legit') {
        document.getElementById('btnLegit').classList.add('active');
        // Auto-fill the Amount from the legit profile
        txAmountInput.value = legitData.Amount.toFixed(2);
    } else {
        document.getElementById('btnFraud').classList.add('active');
        // Auto-fill the Amount from the fraud profile
        txAmountInput.value = fraudData.Amount.toFixed(2);
    }

    // Refresh Payload View
    updateJSONPayloadView();

    // Reset Results UI and clear any stale validation errors
    clearAmountError();
    resetResultUI();
}

function buildPayloadFromUI() {
    const baseData = currentScenario === 'legit' ? { ...legitData } : { ...fraudData };
    const amountVal = parseFloat(txAmountInput.value);
    baseData['Amount'] = isNaN(amountVal) ? 0.0 : amountVal;

    // Ordered keys: Time, V1-V28, Amount  (matches API schema)
    const orderedData = { "Time": baseData["Time"] };
    for (let i = 1; i <= 28; i++) {
        orderedData[`V${i}`] = baseData[`V${i}`];
    }
    orderedData["Amount"] = baseData["Amount"];
    return orderedData;
}

function updateJSONPayloadView() {
    const orderedData = buildPayloadFromUI();
    jsonPayloadTextarea.value = JSON.stringify(orderedData, null, 2);
}


// ----------------------------------------------------
// INPUT VALIDATION
// ----------------------------------------------------
function validateAmount() {
    const raw = txAmountInput.value.trim();

    if (raw === '') {
        showAmountError('Amount is required.');
        return false;
    }

    const val = parseFloat(raw);

    if (isNaN(val)) {
        showAmountError('Amount must be a valid number.');
        return false;
    }

    if (val < AMOUNT_MIN || val > AMOUNT_MAX) {
        showAmountError(`Amount must be between $${AMOUNT_MIN} and $${AMOUNT_MAX.toLocaleString()}.`);
        return false;
    }

    clearAmountError();
    return true;
}

function showAmountError(msg) {
    const el = document.getElementById('amountError');
    el.textContent = msg;
    el.classList.remove('hidden');
    txAmountInput.closest('.input-wrapper').classList.add('input-error');
}

function clearAmountError() {
    const el = document.getElementById('amountError');
    el.textContent = '';
    el.classList.add('hidden');
    txAmountInput.closest('.input-wrapper').classList.remove('input-error');
}


// ----------------------------------------------------
// DEV MODE TOGGLE
// ----------------------------------------------------
function toggleDevMode() {
    isDevMode = !isDevMode;
    const panel = document.getElementById('devPanel');
    const btn = document.getElementById('btnToggleJSON');

    if (isDevMode) {
        panel.classList.remove('hidden');
        btn.textContent = '− Close Developer Mode';
    } else {
        panel.classList.add('hidden');
        btn.textContent = '+ Advanced / JSON Developer Mode';
    }
}


// ----------------------------------------------------
// PREDICTION EXECUTION
// ----------------------------------------------------
async function runPrediction() {
    // Validate amount before sending
    if (!validateAmount()) {
        return;
    }

    const payload = buildPayloadFromUI();
    await executeAPIRequest(payload);
}

async function runManualPrediction() {
    document.getElementById('errorMsg').classList.add('hidden');

    const payloadStr = jsonPayloadTextarea.value;
    let payload;

    try {
        payload = JSON.parse(payloadStr);
    } catch (e) {
        showError('Invalid JSON configuration. Please fix syntax.');
        return;
    }

    // Validate Amount inside manual payload too
    if (payload.Amount !== undefined) {
        if (typeof payload.Amount !== 'number' || payload.Amount < AMOUNT_MIN || payload.Amount > AMOUNT_MAX) {
            showError(`Amount must be a number between $${AMOUNT_MIN} and $${AMOUNT_MAX.toLocaleString()}.`);
            return;
        }
    }

    await executeAPIRequest(payload);
}

async function executeAPIRequest(payload) {
    console.log('[FraudGuard] Sending payload to API:', JSON.stringify(payload, null, 2));
    setLoadingState(true);

    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errBody = await response.text();
            throw new Error(`API Error ${response.status}: ${errBody}`);
        }

        const data = await response.json();

        // Artificial delay so shimmer animation is visible (UX polish)
        setTimeout(() => {
            setLoadingState(false);
            processResult(data);
        }, 800);

    } catch (error) {
        console.error('Fetch Exception:', error);
        setLoadingState(false);
        showError('Failed to reach API. Is FastAPI running on port 8000?');
    }
}


// ----------------------------------------------------
// UI UPDATE LOGIC
// ----------------------------------------------------
function setLoadingState(isLoading) {
    const card = document.getElementById('creditCard');
    const spinner = document.getElementById('spinner');
    const statusText = document.querySelector('.status-text');
    const predictBtn = document.querySelector('.predict-btn');

    resetResultUI();

    if (isLoading) {
        card.classList.add('loading');
        spinner.classList.remove('hidden');
        statusText.innerText = 'Analyzing Transaction...';
        predictBtn.disabled = true;
        predictBtn.style.opacity = '0.6';
        predictBtn.style.cursor = 'not-allowed';
    } else {
        card.classList.remove('loading');
        spinner.classList.add('hidden');
        predictBtn.disabled = false;
        predictBtn.style.opacity = '1';
        predictBtn.style.cursor = 'pointer';
    }
}

function processResult(data) {
    // Expected: { "fraud_probability": 0.001, "prediction": 0, "threshold_used": 0.5 }
    const probValue = data.fraud_probability;
    const isFraud = data.prediction === 1;

    // Show Probability box & Verdict banner
    document.getElementById('probBox').classList.remove('hidden');
    const verdictBanner = document.getElementById('verdictBanner');
    verdictBanner.classList.remove('hidden');

    // Update Indicators
    const indicator = document.getElementById('statusIndicator');
    indicator.className = 'status-indicator'; // reset

    const statusText = document.querySelector('.status-text');
    const verdictText = document.getElementById('verdictText');
    const gaugeFill = document.getElementById('gaugeFill');

    // Format probability (e.g. 0.0015 -> 0.15%)
    const percentage = (probValue * 100).toFixed(2);

    // Animate gauge
    animateValue('probValue', 0, percentage, 800);
    setTimeout(() => {
        gaugeFill.style.width = `${Math.min(percentage, 100)}%`;
    }, 100);

    if (isFraud) {
        indicator.classList.add('danger');
        statusText.innerText = 'Transaction Blocked';
        verdictBanner.classList.add('danger');
        verdictText.innerText = 'FRAUD DETECTED';
        gaugeFill.style.backgroundColor = 'var(--status-danger)';
    } else {
        indicator.classList.add('safe');
        statusText.innerText = 'Transaction Authorized';
        verdictBanner.classList.add('safe');
        verdictText.innerText = 'LEGITIMATE';
        gaugeFill.style.backgroundColor = 'var(--status-safe)';
    }
}

function resetResultUI() {
    const indicator = document.getElementById('statusIndicator');
    indicator.className = 'status-indicator';
    document.querySelector('.status-text').innerText = 'Awaiting Transaction...';

    document.getElementById('probBox').classList.add('hidden');
    document.getElementById('verdictBanner').className = 'verdict-banner hidden';
    document.getElementById('errorMsg').classList.add('hidden');
    document.getElementById('gaugeFill').style.width = '0%';
    document.getElementById('probValue').innerText = '--%';
}

function showError(msg) {
    const errorEl = document.getElementById('errorMsg');
    errorEl.innerText = msg;
    errorEl.classList.remove('hidden');
    document.querySelector('.status-text').innerText = 'System Error';
}

// Small helper for number counting animation
function animateValue(id, start, end, duration) {
    const obj = document.getElementById(id);
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const current = (progress * (end - start)).toFixed(2);
        obj.innerHTML = current + '%';
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}
