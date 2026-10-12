package io.tidevec;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal JSON reader/writer so the SDK stays dependency-free.
 *
 * <p>Parsed values are {@code Map<String,Object>}, {@code List<Object>},
 * {@code String}, {@code Double}, {@code Boolean} or {@code null}.
 * {@link #write} accepts those plus other {@code Number}s, {@code float[]}
 * and {@code Iterable}s.
 */
final class Json {
    private Json() {}

    // ---- writing -------------------------------------------------

    static String write(Object v) {
        StringBuilder sb = new StringBuilder();
        write(sb, v);
        return sb.toString();
    }

    private static void write(StringBuilder sb, Object v) {
        if (v == null) {
            sb.append("null");
        } else if (v instanceof String s) {
            quote(sb, s);
        } else if (v instanceof Boolean b) {
            sb.append(b);
        } else if (v instanceof Float f) {
            if (f.isNaN() || f.isInfinite()) throw new IllegalArgumentException("non-finite number");
            sb.append(f);
        } else if (v instanceof Double d) {
            if (d.isNaN() || d.isInfinite()) throw new IllegalArgumentException("non-finite number");
            sb.append(d);
        } else if (v instanceof Number n) {
            sb.append(n);
        } else if (v instanceof float[] arr) {
            sb.append('[');
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(',');
                write(sb, arr[i]);
            }
            sb.append(']');
        } else if (v instanceof Map<?, ?> m) {
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (!first) sb.append(',');
                first = false;
                quote(sb, String.valueOf(e.getKey()));
                sb.append(':');
                write(sb, e.getValue());
            }
            sb.append('}');
        } else if (v instanceof Iterable<?> it) {
            sb.append('[');
            boolean first = true;
            for (Object o : it) {
                if (!first) sb.append(',');
                first = false;
                write(sb, o);
            }
            sb.append(']');
        } else {
            throw new IllegalArgumentException("cannot serialise " + v.getClass());
        }
    }

    private static void quote(StringBuilder sb, String s) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"'  -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                case '\b' -> sb.append("\\b");
                case '\f' -> sb.append("\\f");
                default -> {
                    if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                    else sb.append(c);
                }
            }
        }
        sb.append('"');
    }

    // ---- reading -------------------------------------------------

    static Object parse(String s) {
        Parser p = new Parser(s);
        p.ws();
        Object v = p.value();
        p.ws();
        if (p.i != s.length()) throw p.error("trailing characters");
        return v;
    }

    private static final class Parser {
        private final String s;
        private int i;

        Parser(String s) { this.s = s; }

        IllegalArgumentException error(String msg) {
            return new IllegalArgumentException("invalid JSON at " + i + ": " + msg);
        }

        void ws() {
            while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++;
        }

        Object value() {
            if (i >= s.length()) throw error("unexpected end");
            char c = s.charAt(i);
            return switch (c) {
                case '{' -> object();
                case '[' -> array();
                case '"' -> string();
                case 't' -> literal("true", Boolean.TRUE);
                case 'f' -> literal("false", Boolean.FALSE);
                case 'n' -> literal("null", null);
                default -> number();
            };
        }

        Object literal(String word, Object v) {
            if (!s.startsWith(word, i)) throw error("expected " + word);
            i += word.length();
            return v;
        }

        Map<String, Object> object() {
            Map<String, Object> m = new LinkedHashMap<>();
            i++;
            ws();
            if (peek('}')) { i++; return m; }
            while (true) {
                ws();
                if (!peek('"')) throw error("expected key");
                String k = string();
                ws();
                expect(':');
                ws();
                m.put(k, value());
                ws();
                if (peek(',')) { i++; continue; }
                expect('}');
                return m;
            }
        }

        List<Object> array() {
            List<Object> l = new ArrayList<>();
            i++;
            ws();
            if (peek(']')) { i++; return l; }
            while (true) {
                ws();
                l.add(value());
                ws();
                if (peek(',')) { i++; continue; }
                expect(']');
                return l;
            }
        }

        String string() {
            expect('"');
            StringBuilder sb = new StringBuilder();
            while (true) {
                if (i >= s.length()) throw error("unterminated string");
                char c = s.charAt(i++);
                if (c == '"') return sb.toString();
                if (c != '\\') { sb.append(c); continue; }
                if (i >= s.length()) throw error("bad escape");
                char e = s.charAt(i++);
                switch (e) {
                    case '"', '\\', '/' -> sb.append(e);
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'u' -> {
                        if (i + 4 > s.length()) throw error("bad \\u escape");
                        sb.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                        i += 4;
                    }
                    default -> throw error("bad escape \\" + e);
                }
            }
        }

        Double number() {
            int start = i;
            while (i < s.length() && "+-0123456789.eE".indexOf(s.charAt(i)) >= 0) i++;
            if (start == i) throw error("unexpected '" + s.charAt(i) + "'");
            try {
                return Double.parseDouble(s.substring(start, i));
            } catch (NumberFormatException e) {
                throw error("bad number");
            }
        }

        boolean peek(char c) { return i < s.length() && s.charAt(i) == c; }

        void expect(char c) {
            if (!peek(c)) throw error("expected '" + c + "'");
            i++;
        }
    }
}
