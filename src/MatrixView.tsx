import type { Matrix } from "./types";

/** Cell shade for a probability; sqrt so rare transitions stay visible. */
export const shade = (p: number) => `rgba(26, 26, 26, ${Math.sqrt(p).toFixed(3)})`;

const LEGEND = [0.01, 0.05, 0.1, 0.25, 0.5, 1];

export function MatrixLegend() {
  return (
    <ul className="legend" aria-label="Colour legend">
      {LEGEND.map((p) => (
        <li key={p}>
          <span className="legend__swatch" style={{ background: shade(p) }} />
          {p * 100}%
        </li>
      ))}
    </ul>
  );
}

const display = (m: Matrix, t: string) => (t === m.start ? "(start)" : t === m.end ? "(end)" : t);

export function MatrixView({ matrix }: { matrix: Matrix }) {
  return (
    <div className="matrix" tabIndex={0} aria-label="Transition matrix, scrollable">
      <table>
        <thead>
          <tr>
            <th className="matrix__corner">from ↓ to →</th>
            {matrix.columns.map((c) => (
              <th key={c} scope="col" className="matrix__col">
                <span>{display(matrix, c)}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.rows.map((row) => {
            const probs = new Map(row.cells);
            return (
              <tr key={row.label}>
                <th scope="row" className="matrix__row">
                  {display(matrix, row.label)}
                </th>
                {matrix.columns.map((c, j) => {
                  const p = probs.get(j) ?? 0;
                  return (
                    <td
                      key={c}
                      style={p ? { background: shade(p) } : undefined}
                      title={`${display(matrix, row.label)} → ${display(matrix, c)}: ${(p * 100).toFixed(1)}%`}
                    />
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
