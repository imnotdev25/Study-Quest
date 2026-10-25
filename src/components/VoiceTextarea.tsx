// A textarea with a 🎙 button. Uses the WebKit speech recognizer when the webview has one; otherwise it focuses
// the field and points to macOS Dictation (press 🌐/Fn twice), which types into any text field.
import { useEffect, useRef, useState } from "react";
import { toast } from "../lib/store";
import { DICTATION, OS } from "../lib/platform";

type Rec = { continuous: boolean; interimResults: boolean; lang: string; start(): void; stop(): void;
  onresult: ((e: { resultIndex: number; results: ArrayLike<{ isFinal: boolean; 0: { transcript: string } }> }) => void) | null;
  onend: (() => void) | null; onerror: ((e: { error: string }) => void) | null };

const Recognizer = (window as unknown as { webkitSpeechRecognition?: new () => Rec; SpeechRecognition?: new () => Rec })
  .SpeechRecognition ?? (window as unknown as { webkitSpeechRecognition?: new () => Rec }).webkitSpeechRecognition;

export function VoiceTextarea({ value, onChange, rows = 3, placeholder }: { value: string; onChange: (v: string) => void;
                                                                           rows?: number; placeholder?: string }) {
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const rec = useRef<Rec | null>(null);
  const area = useRef<HTMLTextAreaElement>(null);
  const latest = useRef(value);
  latest.current = value;
  useEffect(() => () => rec.current?.stop(), []);

  const toggle = () => {
    if (listening) { rec.current?.stop(); return; }
    if (!Recognizer) {
      area.current?.focus();
      toast({ kind: "info", ...DICTATION });
      return;
    }
    const r = new Recognizer();
    r.continuous = true; r.interimResults = true; r.lang = navigator.language || "en-US";
    r.onresult = (e) => {
      let finalText = "", live = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const res = e.results[i];
        if (res.isFinal) finalText += res[0].transcript; else live += res[0].transcript;
      }
      if (finalText) {
        const cur = latest.current;
        onChange((cur && !/\s$/.test(cur) ? cur + " " : cur) + finalText.trim());
      }
      setInterim(live);
    };
    r.onerror = (e) => {
      toast({ kind: "error", title: "🎙 Voice typing stopped", body: e.error === "not-allowed"
        ? `Microphone access was denied. Allow it in your system privacy settings. ${DICTATION.body}`
        : `${e.error}. ${DICTATION.body}` });
    };
    r.onend = () => { setListening(false); setInterim(""); };
    rec.current = r;
    try { r.start(); setListening(true); }
    catch { area.current?.focus(); }
  };

  return (
    <div className="voice-box">
      <textarea ref={area} rows={rows} value={value} placeholder={placeholder} onChange={(e) => onChange(e.target.value)} />
      <button type="button" className={`mic ${listening ? "on" : ""}`} onClick={toggle}
        title={listening ? "Stop voice typing" : OS === "mac" ? "Voice typing (or press 🌐 twice for Dictation)" : OS === "windows" ? "Voice typing (or press Win + H)" : "Voice typing"}>
        {listening ? "⏹" : "🎙"}
      </button>
      {listening && <div className="voice-live">🔴 Listening… {interim && <i>{interim}</i>}</div>}
    </div>
  );
}
