<script setup>
defineProps({ store: Object, options: Object, t: Object });
function toggleGroup(event, group, store) {
  if (event.target.checked) store.selectedMetrics = [...new Set([...store.selectedMetrics, ...group.map(item => item.id)])];
  else store.selectedMetrics = store.selectedMetrics.filter(id => !group.some(item => item.id === id));
}
function toggleMetric(event, id, store) {
  if (event.target.checked && !store.selectedMetrics.includes(id)) store.selectedMetrics.push(id);
  else if (!event.target.checked) store.selectedMetrics = store.selectedMetrics.filter(item => item !== id);
}
</script>

<template>
  <section class="card">
    <h2>{{ t.metrics.title }}</h2>
    <div v-for="(group, groupName) in options?.metric_groups || {}" :key="groupName" class="metric-group">
      <label class="metric-group-header"><input type="checkbox" :checked="group.every(item => store.selectedMetrics.includes(item.id))" @change="toggleGroup($event, group, store)" /><span>{{ t.metrics.groups[groupName] || groupName }}</span></label>
      <div class="metric-grid">
        <label v-for="item in group" :key="item.id" class="metric-check"><input type="checkbox" :value="item.id" :checked="store.selectedMetrics.includes(item.id)" @change="toggleMetric($event, item.id, store)" />{{ item.label }}</label>
      </div>
    </div>
    <div class="form-row"><label class="field-label">{{ t.metrics.samples }}</label><input v-model.number="store.numSamples" type="number" min="1" /></div>
    <div class="form-row"><label class="metric-check"><input v-model="store.experiment1" type="checkbox" />{{ t.metrics.experiment }}</label></div>
  </section>
</template>
