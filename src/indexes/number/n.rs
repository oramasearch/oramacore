use axum_openapi3::utoipa;
use axum_openapi3::utoipa::ToSchema;
use serde::{ser::Error, Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum Number {
    I32(#[schema(inline)] i32),
    F32(#[schema(inline)] f32),
}

impl std::fmt::Display for Number {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Number::I32(value) => write!(f, "{}", value),
            Number::F32(value) => write!(f, "{}", value),
        }
    }
}

impl From<i32> for Number {
    fn from(value: i32) -> Self {
        Number::I32(value)
    }
}
impl From<f32> for Number {
    fn from(value: f32) -> Self {
        Number::F32(value)
    }
}
impl TryFrom<&Value> for Number {
    type Error = serde_json::Error;

    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        match value {
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(Number::I32(i as i32))
                } else if let Some(f) = n.as_f64() {
                    Ok(Number::F32(f as f32))
                } else {
                    Err(serde_json::Error::custom("Not a number"))
                }
            }
            _ => Err(serde_json::Error::custom("Not a number")),
        }
    }
}

impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Number::I32(a), Number::I32(b)) => a == b,
            (Number::I32(a), Number::F32(b)) => *a as f32 == *b,
            (Number::F32(a), Number::F32(b)) => {
                // This is against the IEEE 754-2008 standard,
                // But we don't care here.
                if a.is_nan() && b.is_nan() {
                    return true;
                }
                a == b
            }
            (Number::F32(a), Number::I32(b)) => *a == *b as f32,
        }
    }
}
impl Eq for Number {}

impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Number {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // f32 is implemented as "binary32" type defined in IEEE 754-2008
        // So, it means, it can represent also +/- Infinity and NaN
        // Threat NaN as "more" the Infinity
        // See `total_cmp` method in f32
        match (self, other) {
            (Number::I32(a), Number::I32(b)) => a.cmp(b),
            (Number::I32(a), Number::F32(b)) => (*a as f32).total_cmp(b),
            (Number::F32(a), Number::F32(b)) => a.total_cmp(b),
            (Number::F32(a), Number::I32(b)) => a.total_cmp(&(*b as f32)),
        }
    }
}

impl std::ops::Add for Number {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Number::I32(a), Number::I32(b)) => (a + b).into(),
            (Number::I32(a), Number::F32(b)) => (a as f32 + b).into(),
            (Number::F32(a), Number::F32(b)) => (a + b).into(),
            (Number::F32(a), Number::I32(b)) => (a + b as f32).into(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct SerializableNumber(pub Number);

impl Serialize for SerializableNumber {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        use serde::ser::SerializeTuple;

        match &self.0 {
            Number::F32(v) => {
                let mut tuple = serializer.serialize_tuple(2)?;

                if v.is_infinite() && v.is_sign_positive() {
                    tuple.serialize_element(&3_u8)?;
                    return tuple.end();
                }
                if v.is_infinite() && v.is_sign_negative() {
                    tuple.serialize_element(&4_u8)?;
                    return tuple.end();
                }

                tuple.serialize_element(&1_u8)?;
                tuple.serialize_element(v)?;
                tuple.end()
            }
            Number::I32(v) => {
                let mut tuple = serializer.serialize_tuple(2)?;
                tuple.serialize_element(&2_u8)?;
                tuple.serialize_element(v)?;
                tuple.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for SerializableNumber {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{Error, Visitor};

        struct SerializableNumberVisitor;

        impl<'de> Visitor<'de> for SerializableNumberVisitor {
            type Value = SerializableNumber;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(
                    formatter,
                    "a tuple of size 2 consisting of a u64 discriminant and a value"
                )
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let discriminant: u8 = seq
                    .next_element()?
                    .ok_or_else(|| A::Error::invalid_length(0, &self))?;
                match discriminant {
                    1_u8 => {
                        let x = seq
                            .next_element()?
                            .ok_or_else(|| A::Error::invalid_length(1, &self))?;
                        Ok(SerializableNumber(Number::F32(x)))
                    }
                    2 => {
                        let y = seq
                            .next_element()?
                            .ok_or_else(|| A::Error::invalid_length(1, &self))?;
                        Ok(SerializableNumber(Number::I32(y)))
                    }
                    3 => Ok(SerializableNumber(Number::F32(f32::INFINITY))),
                    4 => Ok(SerializableNumber(Number::F32(f32::NEG_INFINITY))),
                    d => Err(A::Error::invalid_value(
                        serde::de::Unexpected::Unsigned(d.into()),
                        &"1, 2, 3, 4",
                    )),
                }
            }
        }

        deserializer.deserialize_tuple(2, SerializableNumberVisitor)
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::*;

    #[test]
    fn test_number_eq() {
        let a = Number::from(1);
        let b = Number::from(1.0);
        assert_eq!(a, b);
        assert!(a == b);
        assert!(a != Number::from(f32::NAN));

        let a = Number::from(-1);
        let b = Number::from(-1.0);
        assert_eq!(a, b);
        assert!(a == b);
        assert!(a != Number::from(f32::NAN));

        let a = Number::from(2);
        let b = Number::from(2.0);
        assert_eq!(a, b);
        assert!(a == b);
        assert!(a != Number::from(f32::NAN));

        let a = Number::from(f32::INFINITY);
        let b = Number::from(f32::NEG_INFINITY);
        assert_ne!(a, b);
        assert!(a != b);
        assert!(a != Number::from(f32::NAN));

        let a = Number::from(f32::NAN);
        assert_eq!(a, Number::from(f32::NAN));
        assert!(a == Number::from(f32::NAN));
        assert!(a != Number::from(f32::INFINITY));
        assert!(a != Number::from(f32::NEG_INFINITY));
        assert!(a != Number::from(0));
    }

    #[test]
    fn test_number_ord() {
        let a = Number::from(1);

        let b = Number::from(1.0);
        assert_eq!(a.cmp(&b), Ordering::Equal);

        let b = Number::from(2.0);
        assert_eq!(a.cmp(&b), Ordering::Less);
        let b = Number::from(2);
        assert_eq!(a.cmp(&b), Ordering::Less);

        let b = Number::from(-2.0);
        assert_eq!(a.cmp(&b), Ordering::Greater);
        let b = Number::from(-2);
        assert_eq!(a.cmp(&b), Ordering::Greater);

        let a = Number::from(-1);

        let b = Number::from(-1.0);
        assert_eq!(a.cmp(&b), Ordering::Equal);

        let b = Number::from(2.0);
        assert_eq!(a.cmp(&b), Ordering::Less);
        let b = Number::from(2);
        assert_eq!(a.cmp(&b), Ordering::Less);

        let b = Number::from(-2.0);
        assert_eq!(a.cmp(&b), Ordering::Greater);
        let b = Number::from(-2);
        assert_eq!(a.cmp(&b), Ordering::Greater);

        let a = Number::from(1.0);

        let b = Number::from(1.0);
        assert_eq!(a.cmp(&b), Ordering::Equal);

        let b = Number::from(2.0);
        assert_eq!(a.cmp(&b), Ordering::Less);
        let b = Number::from(2);
        assert_eq!(a.cmp(&b), Ordering::Less);

        let b = Number::from(-2.0);
        assert_eq!(a.cmp(&b), Ordering::Greater);
        let b = Number::from(-2);
        assert_eq!(a.cmp(&b), Ordering::Greater);

        let a = Number::from(1);
        assert!(a < Number::from(f32::INFINITY));
        assert!(a > Number::from(f32::NEG_INFINITY));
        assert!(a < Number::from(f32::NAN));

        let a = Number::from(1.0);
        assert!(a < Number::from(f32::INFINITY));
        assert!(a > Number::from(f32::NEG_INFINITY));
        assert!(a < Number::from(f32::NAN));

        let a = Number::from(f32::NAN);
        assert!(a > Number::from(f32::INFINITY));
        assert!(a > Number::from(f32::NEG_INFINITY));
        assert!(a == Number::from(f32::NAN));

        let v = [
            Number::from(1),
            Number::from(1.0),
            Number::from(2),
            Number::from(2.0),
            Number::from(-1),
            Number::from(-1.0),
            Number::from(-2),
            Number::from(-2.0),
            Number::from(f32::INFINITY),
            Number::from(f32::NEG_INFINITY),
            Number::from(f32::NAN),
        ];

        for i in 0..v.len() {
            for j in 0..v.len() {
                let way = v[i].cmp(&v[j]);
                let other_way = v[j].cmp(&v[i]);

                assert_eq!(way.reverse(), other_way);
            }
        }
    }
}
