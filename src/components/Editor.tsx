// CodeMirror 6 Python editor. Cmd/Ctrl+Enter runs.
import { useEffect, useRef } from "react";
import { EditorView, basicSetup } from "codemirror";
import { keymap } from "@codemirror/view";
import { Prec } from "@codemirror/state";
import { python } from "@codemirror/lang-python";
import { oneDark } from "@codemirror/theme-one-dark";

export function Editor({ value, onChange, onRun, docKey }: { value: string; onChange: (v: string) => void;
                                                              onRun: () => void; docKey: string }) {
  const host = useRef<HTMLDivElement>(null);
  const run = useRef(onRun);
  const change = useRef(onChange);
  run.current = onRun;
  change.current = onChange;
  useEffect(() => {
    const view = new EditorView({
      doc: value,
      parent: host.current!,
      extensions: [
        basicSetup, python(), oneDark,
        Prec.highest(keymap.of([{ key: "Mod-Enter", run: () => (run.current(), true) }])),
        EditorView.updateListener.of((u) => { if (u.docChanged) change.current(u.state.doc.toString()); }),
        EditorView.theme({ "&": { height: "100%", fontSize: "13px" }, ".cm-scroller": { fontFamily: "ui-monospace, SF Mono, Menlo, monospace" } }),
      ],
    });
    return () => view.destroy();
    // Recreate only when the exercise (docKey) changes; edits flow out through onChange.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docKey]);
  return <div className="editor" ref={host} />;
}
