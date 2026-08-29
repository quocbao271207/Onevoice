import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_color):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=120, right=120):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_proposal_docx(filename):
    doc = Document()
    
    # Page setup - Normal margins
    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    # Colors
    PRIMARY_COLOR = RGBColor(15, 44, 89)     # Deep Navy
    SECONDARY_COLOR = RGBColor(26, 115, 232) # Vibrant Blue
    DARK_TEXT = RGBColor(33, 37, 41)         # Off-black
    
    # Title / Header
    p_pre = doc.add_paragraph()
    r_pre = p_pre.add_run("ONEVOICE AI CHALLENGE 2026 — TECHNICAL PROPOSAL (PHASE 2)")
    r_pre.font.name = "Arial"
    r_pre.font.size = Pt(10)
    r_pre.font.bold = True
    r_pre.font.color.rgb = SECONDARY_COLOR
    p_pre.paragraph_format.space_after = Pt(4)

    p_title = doc.add_paragraph()
    r_title = p_title.add_run("MediVoice Edge: Ultra-Low Latency, 100% On-Device Medical Speech-to-Speech Translation on Qualcomm Hexagon NPU")
    r_title.font.name = "Arial"
    r_title.font.size = Pt(16)
    r_title.font.bold = True
    r_title.font.color.rgb = PRIMARY_COLOR
    p_title.paragraph_format.space_after = Pt(10)

    # Meta Table
    meta_data = [
        ("Team / Project Name", "MediVoice Edge (Healthcare Edge AI Team)"),
        ("Submission Date", "21 / 08 / 2026 (Phase 2 Technical Submission)"),
        ("Version", "v1.0 (Comprehensive Technical Proposal)"),
        ("Confidentiality", "Restricted — OneVoice AI Challenge Review Only")
    ]
    meta_table = doc.add_table(rows=len(meta_data), cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (label, val) in enumerate(meta_data):
        row = meta_table.rows[idx]
        cell_0, cell_1 = row.cells[0], row.cells[1]
        cell_0.width = Inches(2.2)
        cell_1.width = Inches(4.6)
        set_cell_background(cell_0, "F1F5F9")
        set_cell_margins(cell_0, 60, 60, 100, 100)
        set_cell_margins(cell_1, 60, 60, 100, 100)
        
        p0 = cell_0.paragraphs[0]
        r0 = p0.add_run(label)
        r0.font.name = "Arial"
        r0.font.size = Pt(9)
        r0.font.bold = True
        r0.font.color.rgb = PRIMARY_COLOR
        p0.paragraph_format.space_after = Pt(0)
        
        p1 = cell_1.paragraphs[0]
        r1 = p1.add_run(val)
        r1.font.name = "Arial"
        r1.font.size = Pt(9)
        r1.font.color.rgb = DARK_TEXT
        p1.paragraph_format.space_after = Pt(0)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    def add_heading_1(text):
        h = doc.add_paragraph()
        r = h.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = PRIMARY_COLOR
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(3)
        return h

    def add_heading_2(text):
        h = doc.add_paragraph()
        r = h.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(10.5)
        r.font.bold = True
        r.font.color.rgb = SECONDARY_COLOR
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(2)
        return h

    def add_body(text, bold_prefix=None, space_after=3):
        p = doc.add_paragraph()
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.font.name = "Arial"
            rb.font.size = Pt(9)
            rb.font.bold = True
            rb.font.color.rgb = DARK_TEXT
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(9)
        r.font.color.rgb = DARK_TEXT
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        return p

    def build_table(col_widths, headers, rows_data):
        table = doc.add_table(rows=len(rows_data) + 1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        
        # Header Row
        hdr_row = table.rows[0]
        for idx, title in enumerate(headers):
            cell = hdr_row.cells[idx]
            cell.width = col_widths[idx]
            set_cell_background(cell, "0F2C59")
            set_cell_margins(cell, 80, 80, 100, 100)
            p = cell.paragraphs[0]
            r = p.add_run(title)
            r.font.name = "Arial"
            r.font.size = Pt(8.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)
            p.paragraph_format.space_after = Pt(0)
            
        # Data Rows
        for r_idx, r_data in enumerate(rows_data):
            row = table.rows[r_idx + 1]
            bg_color = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
            for c_idx, val in enumerate(r_data):
                cell = row.cells[c_idx]
                cell.width = col_widths[c_idx]
                set_cell_background(cell, bg_color)
                set_cell_margins(cell, 60, 60, 80, 80)
                p = cell.paragraphs[0]
                r = p.add_run(str(val))
                r.font.name = "Arial"
                r.font.size = Pt(8)
                r.font.color.rgb = DARK_TEXT
                p.paragraph_format.space_after = Pt(0)
        
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # 1. Executive Summary
    add_heading_1("1. Executive Summary")
    add_body("Cross-border healthcare in emergency wards, surgery rooms, and quarantine clinics faces catastrophic verbal communication barriers. Existing translation systems fail in clinical settings because they rely on cloud servers (unusable in shielded wards with zero connectivity), suffer high turnaround delays (>4–6s), violate patient privacy regulations (HIPAA / local medical compliance), and fail to decode bilingual medical code-switching and technical jargon.", bold_prefix="1.1 Problem Overview: ")
    add_body("MediVoice Edge is a ruggedized, wearable, 100% on-device speech-to-speech translation device engineered for high-stakes clinical environments, powered by the Qualcomm Snapdragon QCS6490 platform. Running entirely on local Hexagon NPU cores with zero cloud dependencies, it delivers sub-1.4s end-to-end translation latency (RTF < 0.55) across Vietnamese and English clinical dialogues.", bold_prefix="1.2 Proposed Solution: ")
    add_body("MediVoice Edge delivers four foundational pillars: (1) Absolute Data Privacy & Zero Data Egress ensuring full medical regulatory compliance; (2) Sub-1.4s End-to-End Latency via streaming audio chunking, quantized ASR, and speculative LLM decoding; (3) Specialized Clinical & Code-Switching Accuracy fine-tuned on VietMed, ViMedCSS, MedEV, and FutureBeeAI datasets; and (4) Hands-Free Ergonomics with a 3-microphone beamforming array optimized for high-noise (75+ dB) emergency departments.", bold_prefix="1.3 Key Value Proposition: ")

    # 2. Problem Definition & Target Users
    add_heading_1("2. Problem Definition & Target Users")
    add_heading_2("2.1 Problem Statement")
    add_body("In emergency rooms (ER), intensive care units (ICU), and international field missions, real-time verbal communication directly dictates patient survival. When foreign specialists consult Vietnamese patients or local nurses, language barriers lead to severe clinical complications, miscalculated medication dosages, and delayed surgical interventions. Current solutions (Google Translate, handheld cloud translators) cannot operate in shielded hospital basements, introduce dangerous multi-second latency, leak sensitive medical conversations over public clouds, and collapse when confronted with Vietnamese-English medical code-switching (e.g., mixing clinical terms like 'shock phản vệ', 'test PCR', 'intubation').")

    add_heading_2("2.2 Impact Analysis")
    build_table(
        [Inches(1.8), Inches(2.4), Inches(2.6)],
        ["Impact Area", "Current Pain Point", "Clinical & Operational Consequence"],
        [
            ["Safety & Clinical Operations", "Misinterpreted verbal dosages, allergies, and emergency surgical instructions.", "Medical malpractice, adverse drug events (ADEs), fatal diagnostic delays."],
            ["Productivity & Workflow", "Turnaround delays (>4–6s) in cloud apps stall emergency triage procedures.", "Severe workflow bottlenecks, handoff miscommunications between clinical shifts."],
            ["Connectivity & Reliability", "Hospital basements, shielded ORs, and remote field units have zero or unstable internet.", "Cloud-based translation apps become completely unresponsive at critical moments."],
            ["Data Privacy & Compliance", "Patient voice data is streamed to external cloud servers, violating HIPAA/GDPR.", "Breach of patient confidentiality, institutional non-compliance, heavy legal penalties."]
        ]
    )

    add_heading_2("2.3 Target Users & Use Cases")
    build_table(
        [Inches(1.6), Inches(1.2), Inches(2.8), Inches(1.2)],
        ["User Segment", "Language Need", "Primary Context & Workflow", "Priority"],
        [
            ["Local Vietnamese Doctors & Nurses", "VI ↔ EN (Code-Switching)", "Emergency triage, surgical ward handoffs, ICU vitals monitoring.", "Critical"],
            ["Foreign Specialists & Missionaries", "EN ↔ VI", "Clinical diagnosis, complex surgeries, medical exchange programs.", "Critical"],
            ["International Patients & Tourists", "EN ↔ VI", "Hospital admissions, symptom intake, discharge instructions.", "High"],
            ["Paramedics & Field Responders", "VI ↔ EN", "Ambulance transit, disaster relief stations, remote border clinics.", "High"]
        ]
    )

    add_heading_2("2.4 Key Design Constraints")
    build_table(
        [Inches(2.4), Inches(4.4)],
        ["Constraint Dimension", "Target Specification / Engineering Requirement"],
        [
            ["Internet Dependency", "Zero (0%) — 100% fully autonomous on-device inference at runtime."],
            ["End-to-End Latency", "< 1.5 seconds (Turnaround from speech conclusion to translated audio output)."],
            ["Target Deployment Environment", "Clinical wards, ER, OR, ambulances (Acoustic noise levels up to 75–80 dB)."],
            ["Language Pairs & Dialects", "Vietnamese ↔ English (supports Northern/Central/Southern dialects & code-switching)."],
            ["Physical Form Factor & Battery", "Wearable clinical badge/lanyard, IP54 antimicrobial casing, >10 h continuous shift battery."]
        ]
    )

    # 3. Business Solution & Innovation
    add_heading_1("3. Business Solution & Innovation")
    add_heading_2("3.1 Industry Problem & Solution Fit")
    add_body("Existing mobile translation tools (Google Translate, Microsoft Translator) and consumer translation hardware (Pocketalk, Vasco) are fundamentally misaligned with healthcare realities. They suffer from three fatal failure modes: (1) Mandatory Cloud Connectivity, rendering them useless in shielded hospital basements and radiology suites; (2) Extreme Latency (4–7s round-trip), disrupting high-pressure clinical conversations; and (3) Generic Pretrained Vocabularies, causing hallucinated translations on clinical terms, pharmaceutical names, and mixed Vietnamese-English dialogues.")
    add_body("MediVoice Edge directly resolves this industry gap through a vertically integrated edge AI appliance. By deploying distilled, quantized clinical language models directly onto Qualcomm Hexagon NPUs, MediVoice Edge operates completely air-gapped, delivers real-time translation in <1.4s, and preserves precise medical nomenclature through dedicated domain fine-tuning.")

    add_heading_2("3.2 Innovation & Competitive Strengths")
    build_table(
        [Inches(1.6), Inches(2.6), Inches(2.6)],
        ["Dimension", "Existing Solutions (Cloud / Consumer)", "MediVoice Edge (Our Solution)"],
        [
            ["Connectivity", "Requires constant 4G/5G/Wi-Fi; fails in shielded wards & basements.", "100% Fully Offline — zero internet requirement, air-gapped security."],
            ["Turnaround Latency", "> 4–6 seconds round-trip due to network hops & cloud queueing.", "< 1.4 seconds end-to-end via on-device streaming NPU pipeline."],
            ["Domain Accuracy", "Generic models; frequent hallucinations on clinical & drug terminology.", "Fine-tuned on MedEV, VietMed & ViMedCSS; custom G2P drug lexicon."],
            ["Data Privacy", "Audio & transcripts transmitted to third-party cloud servers.", "100% On-Device execution; zero audio egress, fully HIPAA-compliant."],
            ["Acoustic Noise", "Single-mic consumer input; degrades in loud hospital environments.", "3-mic beamforming array + RNNoise suppression robust to 75+ dB noise."]
        ]
    )
    add_body("MediVoice Edge is the first medical-grade, air-gapped speech translation device engineered specifically for emergency clinical workflows, pairing knowledge-distilled multilingual SLMs with Qualcomm's energy-efficient Hexagon NPU to achieve sub-1.4s turnaround latency at zero connectivity.")

    # 4. AI Approach & Technical Design
    add_heading_1("4. AI Approach & Technical Design")
    add_heading_2("4.1 System Pipeline Overview")
    add_body("MediVoice Edge implements a high-throughput, cascaded streaming speech-to-speech translation pipeline. The system bypasses monolithic end-to-end speech models in favor of an optimized modular architecture, enabling granular latency management, domain-specific adaptation, and hardware acceleration on Qualcomm NPU cores.")
    add_body("Pipeline Stages: Audio Capture (3-Mic Array) → Audio Front-End & VAD (RNNoise + Silero-VAD) → Streaming ASR (Chunked Distil-Whisper INT8) → Clinical Machine Translation (Knowledge-Distilled Llama-3.2-1B / Qwen2.5 INT4) → Streaming Neural TTS (Piper / VITS ONNX-HTP) → Audio Playback & Text OLED.")

    add_heading_2("4.2 Module-by-Module Design")
    build_table(
        [Inches(1.2), Inches(1.8), Inches(0.9), Inches(1.1), Inches(1.8)],
        ["Module", "Model / Framework", "Size (Est.)", "Latency Target", "Key Optimization Technique"],
        [
            ["VAD & Noise Suppression", "Silero-VAD v4 + RNNoise (WebRTC-based)", "~1.5 MB", "< 10 ms / frame", "Dual-mic beamforming, adaptive spectral gating, streaming chunking."],
            ["ASR (Speech Recognition)", "Distil-Whisper-Small.en + Distil-Whisper-Vi (Custom fine-tuned)", "~340 MB (Combined INT8)", "< 380 ms (500ms chunk)", "INT8 QNN compilation, CTC/Attention hybrid decoding, VietMed-tuned."],
            ["NMT / Clinical SLM", "Llama-3.2-1B-Medical / Qwen2.5-0.5B-Medical", "~680 MB (INT4 AWQ)", "< 520 ms (first token <120ms)", "4-bit AWQ weight quant, speculative decoding, medical lexicon injection."],
            ["TTS (Voice Synthesis)", "Piper-TTS / VITS-Edge (Vietnamese & English)", "~65 MB (ONNX-QNN)", "< 280 ms (RTF 0.18)", "INT8 HTP vectorization, mel-spectrogram streaming vocoder."]
        ]
    )

    add_heading_2("4.3 On-Device Optimisation & Memory Management")
    add_body("To execute three neural models within an 8 GB edge envelope without thermal throttling, we employ a rigorous multi-tier optimization strategy:")
    add_body("1. Quantization & Compilation via Qualcomm AI Hub: Models are converted to ONNX, quantized using Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ) to INT8 (ASR, TTS) and INT4 AWQ (SLM), and compiled into Deep Learning Container (.dlc) binaries for Qualcomm Hexagon Tensor Processor (HTP).")
    add_body("2. Zero-Copy Tensor Memory Architecture: Shared ION / DMA memory buffers allow audio spectrograms and token embeddings to pass between DSP, NPU, and CPU without inter-process memory duplication.")
    add_body("3. Memory Footprint Breakdown: Resident ASR (~340 MB) + SLM (~680 MB) + TTS (~65 MB) + Audio Buffers/OS (~1.05 GB) = Total system RAM consumption of ~2.14 GB, comfortably operating within the 8 GB LPDDR4x/5 memory pool.")

    add_heading_2("4.4 Robustness & Edge Case Handling")
    add_body("1. Acoustic Robustness: Trained with synthetic noise injection (hospital alarms, patient monitors, ventilator babble from ESC-50 and AudioSet at 5–15 dB SNR) paired with a 3-mic hardware beamforming array.")
    add_body("2. Linguistic & Code-Switching Handling: Tokenizer vocabulary expanded with 2,400+ ICD-10 medical terms and common bilingual clinical code-switching patterns (ViMedCSS dataset). A custom Grapheme-to-Phoneme (G2P) phonetic mapper handles localized pronunciation of foreign drug brand names (e.g., 'Paracetamol', 'Panadol').")
    add_body("3. Edge Case Fail-safes: Implements low-confidence phrase re-prompting, anti-hallucination repetition penalties, and a sub-50ms instant flash-cache for 100+ standard critical emergency phrases (e.g., 'Check pulse', 'Allergic reaction').")

    # 5. Hardware & Device Concept
    add_heading_1("5. Hardware & Device Concept")
    add_heading_2("5.1 Platform Selection & Justification")
    build_table(
        [Inches(2.0), Inches(1.6), Inches(1.6), Inches(1.6)],
        ["Evaluation Criteria", "Qualcomm QCS6490 (RB3 Gen 2)", "NVIDIA Jetson Orin Nano", "Raspberry Pi 5 + Hailo-8"],
        [
            ["NPU Performance", "12–13 TOPS (Hexagon HTP)", "20–40 TOPS (Ampere GPU)", "13–26 TOPS (Hailo NPU)"],
            ["Thermal Design Power (TDP)", "5.0 – 7.0 W (Selected: 5.2 W peak)", "10.0 – 15.0 W (High heat)", "8.0 – 12.0 W (Moderate)"],
            ["Form Factor & Weight", "Ultra-compact / Wearable badge (<150g)", "Bulky / Heavy heatsink (>350g)", "Modular / Fragile assembly (>220g)"],
            ["SDK & Edge Toolchain", "Qualcomm QNN, SNPE, AI Hub (Selected)", "NVIDIA TensorRT", "Hailo TAPPAS / PyTorch ONNX"]
        ]
    )
    add_body("Platform Selection Justification: The Qualcomm Snapdragon QCS6490 (RB3 Gen 2) was selected because it provides the industry's highest NPU energy efficiency (13 TOPS at <5.5W peak power), native INT4/INT8 Hexagon tensor acceleration, integrated audio DSP, and an ultra-compact footprint ideal for an antimicrobial, wearable clinical badge (<150g, IP54).")

    add_heading_2("5.2 Key Hardware Components & Power Budget")
    build_table(
        [Inches(1.8), Inches(2.2), Inches(1.2), Inches(1.6)],
        ["Component", "Technical Specification", "Peak Power", "Operational Notes"],
        [
            ["SoC / Compute Core", "Qualcomm QCS6490 (Kryo CPU, Adreno GPU, Hexagon NPU)", "3.8 W", "Active NPU + CPU concurrent inference"],
            ["Microphone Array", "3× MEMS Omnidirectional Microphones (16 kHz, 64 dB SNR)", "0.08 W", "Hardware beamforming & noise cancellation"],
            ["Audio Output / Amp", "Class-D Smart Audio Amplifier + 1.5W Speaker / BT 5.3", "0.65 W", "Crisp voice playback; BLE 5.3 headset option"],
            ["Display (Verification)", "2.13-inch Sunlight-Readable Monochrome E-Paper / Low-Power OLED", "0.15 W", "Displays real-time translated text verification"],
            ["System Memory & Flash", "8 GB LPDDR4x + 64 GB UFS 3.1", "0.35 W", "Full on-device OS & pre-compiled DLC storage"],
            ["Battery Subsystem", "3,500 mAh 3.8V High-Density Li-Po Cell", "—", "> 10.5 hours active clinical shift runtime"],
            ["Total System Peak Budget", "All modules actively computing & speaking", "5.03 W", "Standby / Listening average: ~0.85 W"]
        ]
    )

    # 6. System Architecture & Integration
    add_heading_1("6. System Architecture & Integration")
    add_heading_2("6.1 Software Stack")
    build_table(
        [Inches(1.5), Inches(2.5), Inches(2.8)],
        ["Software Layer", "Framework / Technology Component", "Functional Role in MediVoice Edge"],
        [
            ["Application & UX Layer", "C++20 Core Orchestrator + Embedded Qt/QML UI", "Event routing, state machine, PTT button handling, OLED display."],
            ["Inference Runtime", "Qualcomm Neural Processing SDK (QNN) & ONNX EP", "Executes quantized ASR, MT, and TTS models on Hexagon NPU."],
            ["Audio DSP & Front-End", "TinyALSA + WebRTC AudioProcessing / RNNoise", "Real-time acoustic echo cancellation, beamforming, and VAD."],
            ["Model Serving & Cache", "Zero-Copy Circular Ring-Buffer & Flash Cache", "Streaming token passing and instant phrase replay (<50ms)."],
            ["Operating System", "Custom Yocto Linux (Kernel 6.1 LTS with RT-patches)", "Deterministic low-latency audio scheduling and power management."]
        ]
    )

    add_heading_2("6.2 Offline-First Design Principles")
    add_body("1. Zero Network Dependency: 100% of neural network weights, lexicons, and audio decoders are baked into the on-board 64 GB UFS 3.1 storage. The device has no telemetry egress, making it inherently air-gapped.")
    add_body("2. Pre-compiled NPU Execution: All models are pre-compiled into Qualcomm DLC binaries during build time, eliminating runtime compilation overhead and ensuring predictable memory allocation.")
    add_body("3. Deterministic Low Latency: Audio frames (16 kHz, 16-bit mono) are processed in 500ms sliding windows, enabling the translation engine to begin decoding before the speaker finishes long sentences.")

    # 7. Team Profile & Project Timeline
    add_heading_1("7. Team Profile & Project Timeline")
    add_heading_2("7.1 Team Members")
    build_table(
        [Inches(1.5), Inches(1.6), Inches(1.8), Inches(1.9)],
        ["Member Name", "Assigned Role", "Core Expertise", "Project Contribution Area"],
        [
            ["Quoc Bao Trinh", "Team Lead & AI Architect", "Edge AI, NLP, Speech Processing", "Overall pipeline design, ASR/MT fine-tuning & model distillation."],
            ["Thai Nguyen", "Machine Learning Engineer", "Parallel Corpora, MT Benchmarking", "MedEV & FutureBeeAI dataset curation, BLEU/COMET evaluation."],
            ["Khoa Pham", "Data & Quality Engineer", "Clinical Data Analysis, QA Systems", "ViMedCSS, medical QA integration, deduplication & testing."],
            ["Minh Vo", "Embedded Hardware Engineer", "Embedded Linux, Qualcomm SDK", "QCS6490 hardware integration, QNN quantization & power profiling."]
        ]
    )

    add_heading_2("7.2 Project Timeline & Milestones")
    build_table(
        [Inches(0.8), Inches(1.8), Inches(2.8), Inches(1.4)],
        ["Phase", "Milestone Title", "Key Activities & Deliverables", "Target Date"],
        [
            ["Phase 1", "Requirements & Data EDA", "Domain validation, VietMed/MedEV EDA, Qualcomm AI Hub model benchmarking.", "June 2026 (Completed)"],
            ["Phase 2", "Technical Proposal & Pipeline Spec", "Comprehensive system design, model quantization strategy, BOM & power analysis.", "August 2026 (Current)"],
            ["Phase 3", "On-Device Prototype Integration", "Deployment of DLC models on QCS6490 NPU, streaming C++ pipeline, WER/BLEU tuning.", "September 2026"],
            ["Phase 4", "Field Testing & Grand Finale Demo", "Simulated hospital ER trials, noise robustness verification, final submission.", "October–November 2026"]
        ]
    )

    # 8. Submission Checklist
    add_heading_1("8. Submission Checklist")
    build_table(
        [Inches(0.6), Inches(5.0), Inches(1.2)],
        ["#", "Checklist Item", "Status"],
        [
            ["1", "Executive Summary written (200–300 words covering problem, solution, value prop)", "[X] Complete"],
            ["2", "Problem Statement, Impact Analysis, Target Users, and Design Constraints completed", "[X] Complete"],
            ["3", "Business Solution completed — industry gap, solution fit, and competitive differentiation described", "[X] Complete"],
            ["4", "AI pipeline documented with model specs, latency targets, and on-device optimization strategy", "[X] Complete"],
            ["5", "Hardware platform justified; form factor, BOM, and power budget filled in", "[X] Complete"],
            ["6", "Team profiles and project timeline filled in with feasible milestones", "[X] Complete"],
            ["7", "All placeholder text replaced; architecture, memory, and data flow detailed", "[X] Complete"],
            ["8", "Document exported and formatted for submission portal review", "[X] Complete"]
        ]
    )

    doc.save(filename)
    print(f"Successfully generated proposal document: {filename}")

if __name__ == "__main__":
    create_proposal_docx("docs/Technical_Proposal_Phase2_OneVoice.docx")
