package io.tidevec;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class JsonTest {

    @Test
    void escapesHostileStrings() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("id", "a\"b\\c\n\u0001");
        String s = Json.write(m);
        assertEquals("{\"id\":\"a\\\"b\\\\c\\n\\u0001\"}", s);
        assertEquals(m, Json.parse(s));
    }

    @Test
    void roundTripsNestedValues() {
        Object v = Json.parse("{\"a\":[1,2.5,-3e2,true,false,null],\"b\":{\"c\":\"\\u00e9\"}}");
        Map<?, ?> m = (Map<?, ?>) v;
        assertEquals(List.of(1.0, 2.5, -300.0, true, false), ((List<?>) m.get("a")).subList(0, 5));
        assertNull(((List<?>) m.get("a")).get(5));
        assertEquals("é", ((Map<?, ?>) m.get("b")).get("c"));
        assertEquals(v, Json.parse(Json.write(v)));
    }

    @Test
    void writesFloatArrays() {
        assertEquals("[1.0,-0.5]", Json.write(new float[]{1f, -0.5f}));
    }

    @Test
    void rejectsMalformedInput() {
        assertThrows(IllegalArgumentException.class, () -> Json.parse("{\"a\":}"));
        assertThrows(IllegalArgumentException.class, () -> Json.parse("[1,2"));
        assertThrows(IllegalArgumentException.class, () -> Json.parse("{} x"));
        assertThrows(IllegalArgumentException.class, () -> Json.write(Float.NaN));
    }
}
