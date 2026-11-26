use std::collections::BTreeMap;
pub trait AnemoPlugin: Send + Sync { fn name(&self) -> &str; fn compute(&self, inputs: &[f64]) -> Vec<f64>; }
#[derive(Default)]
pub struct PluginRegistry { plugins: BTreeMap<String, Box<dyn AnemoPlugin>> }
impl PluginRegistry {
    pub fn register<P: AnemoPlugin + 'static>(&mut self, p: P) { self.plugins.insert(p.name().to_string(), Box::new(p)); }
    pub fn execute(&self, name: &str, inputs: &[f64]) -> Option<Vec<f64>> { self.plugins.get(name).map(|p| p.compute(inputs)) }
}
